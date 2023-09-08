pub mod fixup;
pub mod frame;
// pub mod scrub;
pub mod segments;
pub mod templates;
// pub mod test;
pub mod video;

pub use opencv;

use std::sync::atomic::AtomicUsize;

pub static FRAMES_PROCESSED: AtomicUsize = AtomicUsize::new(0);
