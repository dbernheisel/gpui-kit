//! Dispatches GTK's events from GPUI without polling.
//!
//! GPUI owns the main thread's event loop, so GTK's, GLib's default main context, is driven by
//! hand. On the main thread, where GTK runs, the pump dispatches what is ready, then asks GLib
//! what to wait for next: its file descriptors (X11, WebKit's IPC sockets, GLib's own wakeup)
//! and the time until its next timer. A helper thread waits for those in `g_poll` and wakes the
//! pump when one is ready. The main thread owns the context for the app's life, as `gtk_main`
//! would, so GLib signals its wakeup descriptor, which the helper polls, whenever another thread
//! adds a source or moves a timer. GLib doesn't signal for the owner's own sources, so calls
//! into GTK from GPUI nudge the helper; that also covers events GTK queues without making a
//! descriptor ready (an X11 event read ahead of a reply, a synthesized crossing event). Sources
//! the main thread adds while dispatching are seen when the pump asks GLib what to wait for.
//!
//! The pump runs only while a web view shows in a visible window. When the last one hides or
//! is released, it keeps dispatching until GLib goes quiet (WebKit takes a few more turns to
//! hide or tear down a view), for at most [`SETTLE_LIMIT`], then flushes GDK's X11 requests and
//! stops; nothing wakes for GTK until a web view shows again.

use std::{
    cell::Cell,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use gpui::{App, Global};
use gtk::glib::{self, ffi, translate::ToGlibPtr as _};

/// The most main context iterations the pump runs before GPUI gets a turn, so a source that is
/// always ready cannot starve GPUI.
const MAX_ITERATIONS_PER_ROUND: usize = 64;

/// After the last web view stops showing, how long a wait must pass with nothing ready before
/// the pump stops.
const SETTLE_QUIET_MS: i32 = 100;

/// The longest the pump keeps running after the last web view stops showing, so a page that
/// stays busy out of sight cannot keep it awake.
const SETTLE_LIMIT: Duration = Duration::from_secs(2);

static ROUNDS: AtomicU64 = AtomicU64::new(0);

/// How many times the GTK pump has woken to dispatch GTK's events.
///
/// A diagnostic for measuring that the pump stays idle: it does not grow while no web view
/// shows.
#[doc(hidden)]
pub fn rounds() -> u64 {
    ROUNDS.load(Ordering::Relaxed)
}

/// Drives GTK's main loop while a web view shows. Set as a global once GTK is initialized.
pub(crate) struct GtkPump(Rc<PumpState>);

impl Global for GtkPump {}

impl GtkPump {
    /// Start the pump on GLib's default main context. It runs once, then stops until a web
    /// view shows.
    pub(crate) fn start(cx: &mut App) -> std::io::Result<Self> {
        let context = glib::MainContext::default();
        // Owned for good: no moment without an owner, when GLib would wake no one for a
        // source another thread adds.
        std::mem::forget(context.acquire().map_err(std::io::Error::other)?);
        let waiter = Waiter::spawn()?;
        let (state, resumed) = PumpState::new(context);
        let pump = state.clone();
        cx.spawn(async move |_| pump.run(waiter, resumed).await)
            .detach();
        Ok(Self(state))
    }

    /// A web view's vote, counted while the view and its window are both visible.
    pub(crate) fn vote(&self, window_visible: bool) -> Rc<PumpVote> {
        PumpVote::new(self.0.clone(), window_visible)
    }
}

/// Wakes the pump after a call into GTK, without a say in whether it runs.
#[derive(Clone)]
pub(crate) struct PumpNudge(Rc<PumpState>);

impl PumpNudge {
    pub(crate) fn nudge(&self) {
        self.0.nudge();
    }
}

struct PumpState {
    context: glib::MainContext,
    /// How many web views show in a visible window.
    showing: Cell<usize>,
    /// Whether the helper thread is waiting for GLib, so a nudge has someone to wake.
    waiting: Cell<bool>,
    resume: async_channel::Sender<()>,
}

impl PumpState {
    fn new(context: glib::MainContext) -> (Rc<Self>, async_channel::Receiver<()>) {
        let (resume, resumed) = async_channel::bounded(1);
        let state = Rc::new(Self {
            context,
            showing: Cell::new(0),
            waiting: Cell::new(false),
            resume,
        });
        (state, resumed)
    }

    async fn run(&self, waiter: Waiter, resumed: async_channel::Receiver<()>) {
        let mut fds = Vec::new();
        // Since when the pump has been settling, after the last web view stopped showing.
        let mut settling = None;
        let mut quiet = false;
        loop {
            ROUNDS.fetch_add(1, Ordering::Relaxed);
            for _ in 0..MAX_ITERATIONS_PER_ROUND {
                if !self.context.iteration(false) {
                    break;
                }
            }
            let mut wait = WaitSet::query(&self.context, fds);

            if self.showing.get() > 0 {
                settling = None;
            } else if (quiet && wait.timeout != 0)
                || settling.get_or_insert_with(Instant::now).elapsed() >= SETTLE_LIMIT
            {
                // Send the X11 requests GTK has buffered, such as unmapping the view that just
                // hid, since nothing flushes them while the pump is stopped.
                if let Some(display) = gtk::gdk::Display::default() {
                    display.flush();
                }
                fds = wait.fds;
                while self.showing.get() == 0 {
                    if resumed.recv().await.is_err() {
                        return;
                    }
                }
                settling = None;
                quiet = false;
                continue;
            }

            // While settling, GLib has gone quiet once a wait lasts `SETTLE_QUIET_MS` with
            // nothing ready, and nothing is ready after the round that follows it.
            let capped = settling.is_some() && (wait.timeout < 0 || wait.timeout > SETTLE_QUIET_MS);
            if capped {
                wait.timeout = SETTLE_QUIET_MS;
            }
            self.waiting.set(true);
            let wait = waiter.wait(wait).await;
            self.waiting.set(false);
            let Some(wait) = wait else {
                return;
            };
            quiet = capped && !wait.ready;
            fds = wait.fds;
        }
    }

    fn show(&self) {
        let showing = self.showing.get();
        self.showing.set(showing + 1);
        if showing == 0 {
            // Full means a resume is already queued.
            let _ = self.resume.try_send(());
        }
    }

    fn hide(&self) {
        let showing = self.showing.get() - 1;
        self.showing.set(showing);
        if showing == 0 {
            // Return from the wait, so the pump dispatches what is pending and stops.
            self.nudge();
        }
    }

    /// Wake the pump after a call into GTK from GPUI. Calls before the next wait share one
    /// wakeup.
    fn nudge(&self) {
        if self.waiting.replace(false) {
            self.context.wakeup();
        }
    }
}

/// One web view's say in whether the pump runs: it counts while the view is shown in a
/// visible window.
pub(crate) struct PumpVote {
    pump: Rc<PumpState>,
    view: Cell<bool>,
    window: Cell<bool>,
    counted: Cell<bool>,
}

impl PumpVote {
    fn new(pump: Rc<PumpState>, window_visible: bool) -> Rc<Self> {
        // Wry builds its view visible.
        let vote = Rc::new(Self {
            pump,
            view: Cell::new(true),
            window: Cell::new(window_visible),
            counted: Cell::new(false),
        });
        vote.update();
        vote
    }

    pub(crate) fn set_view_visible(&self, visible: bool) {
        self.view.set(visible);
        self.update();
    }

    pub(crate) fn set_window_visible(&self, visible: bool) {
        self.window.set(visible);
        self.update();
    }

    /// Wake the pump after calling into GTK, which can add sources or queue GTK events without
    /// making one of GLib's file descriptors ready.
    pub(crate) fn nudge(&self) {
        self.pump.nudge();
    }

    /// A nudge for a holder of this web view that has no vote, such as a `WebViewHandle`.
    pub(crate) fn nudger(&self) -> PumpNudge {
        PumpNudge(self.pump.clone())
    }

    fn update(&self) {
        let showing = self.view.get() && self.window.get();
        if self.counted.replace(showing) != showing {
            if showing {
                self.pump.show();
            } else {
                self.pump.hide();
            }
        }
    }
}

impl Drop for PumpVote {
    fn drop(&mut self) {
        if self.counted.get() {
            self.pump.hide();
        }
    }
}

/// What GLib waits for next: its file descriptors, and the milliseconds until its next timer
/// (`-1` for none).
struct WaitSet {
    fds: Vec<ffi::GPollFD>,
    timeout: i32,
    /// Whether the wait ended with a file descriptor ready, rather than at the timeout.
    ready: bool,
}

impl WaitSet {
    /// Ask `context`, which the caller owns, what to wait for.
    ///
    /// This runs GLib's prepare, query and check steps without dispatching, as
    /// `g_main_context_pending` does, so the context is never left between steps while GPUI
    /// runs. The next iteration prepares again, which discards the check's results.
    fn query(context: &glib::MainContext, mut fds: Vec<ffi::GPollFD>) -> Self {
        let context = context.to_glib_none().0;
        let mut max_priority = 0;
        let mut timeout = -1;
        // SAFETY: The caller owns `context`, and each count passed with `fds` is its length.
        unsafe {
            let prepared = ffi::g_main_context_prepare(context, &mut max_priority) != ffi::GFALSE;
            loop {
                let count = ffi::g_main_context_query(
                    context,
                    max_priority,
                    &mut timeout,
                    fds.as_mut_ptr(),
                    fds.len() as i32,
                ) as usize;
                if count <= fds.len() {
                    fds.truncate(count);
                    break;
                }
                fds.resize(
                    count,
                    ffi::GPollFD {
                        fd: -1,
                        events: 0,
                        revents: 0,
                    },
                );
            }
            // Nothing has been polled, so only sources that are ready without I/O, such as an
            // expired timer, pass the check.
            for fd in &mut fds {
                fd.revents = 0;
            }
            let ready = ffi::g_main_context_check(
                context,
                max_priority,
                fds.as_mut_ptr(),
                fds.len() as i32,
            ) != ffi::GFALSE;
            if prepared || ready {
                timeout = 0;
            }
        }
        Self {
            fds,
            timeout,
            ready: false,
        }
    }
}

/// The helper thread that waits on GLib's file descriptors for the pump. It only polls: the
/// main thread owns the context throughout.
struct Waiter {
    requests: mpsc::Sender<WaitSet>,
    done: async_channel::Receiver<WaitSet>,
}

impl Waiter {
    fn spawn() -> std::io::Result<Self> {
        let (requests, pending) = mpsc::channel::<WaitSet>();
        let (finished, done) = async_channel::bounded(1);
        thread::Builder::new()
            .name("gtk-wait".into())
            .spawn(move || {
                for mut wait in pending {
                    // SAFETY: `fds` holds `len` descriptors.
                    wait.ready = unsafe {
                        ffi::g_poll(wait.fds.as_mut_ptr(), wait.fds.len() as u32, wait.timeout)
                    } > 0;
                    if finished.send_blocking(wait).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self { requests, done })
    }

    /// Wait on the helper thread until something in `wait` is ready, and hand its buffer back.
    async fn wait(&self, wait: WaitSet) -> Option<WaitSet> {
        self.requests.send(wait).ok()?;
        self.done.recv().await.ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::AtomicBool};

    /// Run `wait` on the helper as the pump does, without GPUI's executor.
    fn wait_blocking(waiter: &Waiter, wait: WaitSet) -> WaitSet {
        waiter.requests.send(wait).unwrap();
        waiter.done.recv_blocking().unwrap()
    }

    /// Ask GLib what to wait for. The tests that wait hold ownership on the test thread
    /// throughout, as the main thread does; acquiring again here is recursive.
    fn query(context: &glib::MainContext) -> WaitSet {
        let _owner = context.acquire().unwrap();
        WaitSet::query(context, Vec::new())
    }

    fn idle_source(fired: &Arc<AtomicBool>) -> glib::Source {
        let fired = fired.clone();
        glib::idle_source_new(None, glib::Priority::DEFAULT, move || {
            fired.store(true, Ordering::SeqCst);
            glib::ControlFlow::Break
        })
    }

    /// Attach an idle source from a worker thread, as GDBus or a GTask returning to the
    /// default context does.
    fn attach_from_another_thread(context: &glib::MainContext, fired: &Arc<AtomicBool>) {
        let context = context.clone();
        let fired = fired.clone();
        thread::spawn(move || {
            idle_source(&fired).attach(Some(&context));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn idle_context_waits_without_a_timeout() {
        let context = glib::MainContext::new();
        let wait = query(&context);
        assert_eq!(wait.timeout, -1, "an idle context has no timer to wake for");
        // GLib's own wakeup descriptor.
        assert!(!wait.fds.is_empty());
    }

    #[test]
    fn waits_until_the_next_timer() {
        let context = glib::MainContext::new();
        let _owner = context.acquire().unwrap();
        let fired = Arc::new(AtomicBool::new(false));
        let source =
            glib::timeout_source_new(Duration::from_millis(100), None, glib::Priority::DEFAULT, {
                let fired = fired.clone();
                move || {
                    fired.store(true, Ordering::SeqCst);
                    glib::ControlFlow::Break
                }
            });
        source.attach(Some(&context));

        let wait = query(&context);
        assert!(
            (1..=100).contains(&wait.timeout),
            "timeout {}",
            wait.timeout
        );

        let waiter = Waiter::spawn().unwrap();
        let started = Instant::now();
        let wait = wait_blocking(&waiter, wait);
        assert!(started.elapsed() >= Duration::from_millis(80));
        assert!(!wait.ready, "a timer ends the wait at its timeout");
        while context.iteration(false) {}
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn a_source_another_thread_adds_while_waiting_wakes_the_helper() {
        let context = glib::MainContext::new();
        let _owner = context.acquire().unwrap();
        let waiter = Waiter::spawn().unwrap();
        let wait = query(&context);
        assert_eq!(wait.timeout, -1);

        waiter.requests.send(wait).unwrap();
        // The helper is waiting with no timeout; nothing should come back on its own.
        thread::sleep(Duration::from_millis(200));
        assert!(
            waiter.done.try_recv().is_err(),
            "the helper woke while idle"
        );

        // An idle source touches no descriptor; GLib signals its wakeup because this thread,
        // not the one attaching, owns the context.
        let fired = Arc::new(AtomicBool::new(false));
        let started = Instant::now();
        attach_from_another_thread(&context, &fired);
        let wait = waiter.done.recv_blocking().unwrap();
        assert!(started.elapsed() < Duration::from_millis(100));
        assert!(wait.ready);
        while context.iteration(false) {}
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn a_source_another_thread_adds_before_the_wait_wakes_the_helper() {
        let context = glib::MainContext::new();
        let _owner = context.acquire().unwrap();
        let waiter = Waiter::spawn().unwrap();
        let wait = query(&context);
        assert_eq!(wait.timeout, -1);

        // Between asking GLib what to wait for and the helper's poll.
        let fired = Arc::new(AtomicBool::new(false));
        attach_from_another_thread(&context, &fired);
        waiter.requests.send(wait).unwrap();
        thread::sleep(Duration::from_millis(500));
        assert!(
            waiter.done.try_recv().is_ok_and(|wait| wait.ready),
            "the helper slept through a ready idle source"
        );
        while context.iteration(false) {}
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn a_source_the_owner_adds_waits_for_a_nudge() {
        let context = glib::MainContext::new();
        let _owner = context.acquire().unwrap();
        let waiter = Waiter::spawn().unwrap();
        waiter.requests.send(query(&context)).unwrap();

        // GLib doesn't signal for the owner's own sources, which is why calls into GTK nudge.
        let fired = Arc::new(AtomicBool::new(false));
        idle_source(&fired).attach(Some(&context));
        thread::sleep(Duration::from_millis(200));
        assert!(waiter.done.try_recv().is_err());

        context.wakeup();
        let started = Instant::now();
        let wait = waiter.done.recv_blocking().unwrap();
        assert!(started.elapsed() < Duration::from_millis(100));
        assert!(wait.ready, "GLib's wakeup descriptor ends the wait");
        while context.iteration(false) {}
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn votes_count_views_shown_in_visible_windows() {
        let (pump, resumed) = PumpState::new(glib::MainContext::new());

        let hidden_window = PumpVote::new(pump.clone(), false);
        assert_eq!(pump.showing.get(), 0);
        assert!(resumed.try_recv().is_err());

        let first = PumpVote::new(pump.clone(), true);
        assert_eq!(pump.showing.get(), 1);
        assert!(
            resumed.try_recv().is_ok(),
            "the first view to show resumes the pump"
        );

        hidden_window.set_window_visible(true);
        assert_eq!(pump.showing.get(), 2);
        assert!(
            resumed.try_recv().is_err(),
            "only the first view resumes it"
        );

        first.set_view_visible(false);
        first.set_view_visible(false);
        assert_eq!(pump.showing.get(), 1);
        drop(hidden_window);
        assert_eq!(
            pump.showing.get(),
            0,
            "releasing the last view stops the pump"
        );
        drop(first);
        assert_eq!(pump.showing.get(), 0);

        let again = PumpVote::new(pump.clone(), true);
        assert_eq!(pump.showing.get(), 1);
        assert!(
            resumed.try_recv().is_ok(),
            "building a view again restarts the pump"
        );
        again.set_window_visible(false);
        assert_eq!(pump.showing.get(), 0, "a minimized window stops the pump");
    }

    #[test]
    fn nudges_share_one_wakeup_per_wait() {
        let context = glib::MainContext::new();
        let (pump, _resumed) = PumpState::new(context.clone());
        pump.waiting.set(true);
        pump.nudge();
        assert!(!pump.waiting.get());
        // The wakeup is pending: the next iteration consumes it and dispatches nothing.
        let wait = query(&context);
        assert_eq!(wait.timeout, -1);
        pump.nudge();
        assert!(!pump.waiting.get());
    }
}
