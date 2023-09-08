pub mod cpu;
//#[cfg(feature = "cuda")]
// pub mod gpu;

use color_eyre::eyre::{eyre, Context, Result};
use crossbeam_channel::{Receiver, Sender};
use opencv::{
	core::Mat,
	videoio::{VideoCapture, VideoCaptureTrait},
};

pub struct Frame {
	index: usize,
	frame: Mat,
}

impl Frame {
	#[inline]
	pub fn index(&self) -> usize {
		self.index
	}

	#[inline]
	pub fn frame(&self) -> &Mat {
		&self.frame
	}

	#[inline]
	pub fn into_frame(self) -> Mat {
		self.frame
	}
}

pub type FrameSender = Sender<Frame>;
pub type FrameReceiver = Receiver<Frame>;
pub type MatchedFrameSender = Sender<usize>;
pub type MatchedFrameReceiver = Receiver<usize>;

pub fn send_frames(capture: &mut VideoCapture, frame_sender: FrameSender) -> Result<()> {
	let mut index = 1_usize;
	let mut raw_frame = Mat::default();
	let mut mid_a = Mat::default();
	let mut mid_b = Mat::default();
	while capture
		.read(&mut raw_frame)
		.wrap_err_with(|| format!("failed to read frame {index} from video capture input"))?
	{
		let frame = crate::fixup::fixup_frame(&raw_frame, &mut mid_a, &mut mid_b, true)
			.wrap_err_with(|| format!("failed to fixup image from frame {index}"))?;
		frame_sender
			.send(Frame { index, frame })
			.map_err(|_| eyre!("failed to send frame {index} to worker threads"))?;
		index += 1;
	}
	Ok(())
}
