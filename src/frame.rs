pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;

use crate::FrameSender;
use color_eyre::eyre::{eyre, Context, Result};
use opencv::{
	core::Mat,
	videoio::{VideoCapture, VideoCaptureTrait},
};

pub fn send_frames(capture: &mut VideoCapture, frame_sender: FrameSender) -> Result<()> {
	let mut frame_num = 1_usize;
	loop {
		let mut frame = Mat::default();
		if !capture
			.read(&mut frame)
			.wrap_err("failed to read frame from video capture")?
		{
			break;
		}
		frame_sender
			.send((frame, frame_num))
			.map_err(|_| eyre!("failed to send frame {frame_num} to worker threads"))?;
		frame_num += 1;
	}
	Ok(())
}
