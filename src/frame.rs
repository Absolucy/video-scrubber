pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;

use crate::FrameSender;
use color_eyre::eyre::{eyre, Context, Result};
use opencv::{
	core::Mat,
	imgproc::COLOR_BGR2GRAY,
	videoio::{VideoCapture, VideoCaptureTrait},
};

pub fn send_frames(capture: &mut VideoCapture, frame_sender: FrameSender) -> Result<()> {
	let mut frame_num = 1_usize;
	let mut raw_frame = Mat::default();
	loop {
		if !capture
			.read(&mut raw_frame)
			.wrap_err("failed to read frame from video capture")?
		{
			break;
		}
		let mut grey_frame = Mat::default();
		opencv::imgproc::cvt_color(&raw_frame, &mut grey_frame, COLOR_BGR2GRAY, 0)
			.wrap_err("failed to convert frame to greyscale")?;
		frame_sender
			.send((grey_frame, frame_num))
			.map_err(|_| eyre!("failed to send frame {frame_num} to worker threads"))?;
		frame_num += 1;
	}
	Ok(())
}
