use color_eyre::eyre::{Result, WrapErr};
use opencv::core::{Mat, Size, BORDER_DEFAULT, NORM_MINMAX};

const BLUR_K_SIZE: Size = Size::new(5, 5);

pub fn fixup_frame(base_image: &Mat, mid_a: &mut Mat, mid_b: &mut Mat, grey: bool) -> Result<Mat> {
	let image = if grey {
		opencv::imgproc::cvt_color(base_image, mid_a, opencv::imgproc::COLOR_BGR2GRAY, 0)
			.wrap_err("failed to convert frame to greyscale")?;
		mid_a
	} else {
		base_image
	};
	opencv::imgproc::gaussian_blur(&image, mid_b, BLUR_K_SIZE, 0.0, 0.0, BORDER_DEFAULT)
		.wrap_err("failed to apply gaussian blur")?;
	let mut result = Mat::default();
	opencv::core::normalize(
		mid_b,
		&mut result,
		0.0,
		255.0,
		NORM_MINMAX,
		-1,
		&Mat::default(),
	)
	.wrap_err("failed to normalize image")?;
	Ok(result)
}

pub fn fixup_frame_2(base_image: &Mat, grey: bool) -> Result<Mat> {
	let mut mid_a = Mat::default();
	let mut mid_b = Mat::default();
	fixup_frame(base_image, &mut mid_a, &mut mid_b, grey)
}
