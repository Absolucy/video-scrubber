pub mod cmd;
pub mod scrub;

use self::cmd::{CliArgs, CliSubcommands};
use clap::Parser;
use color_eyre::eyre::Result;

#[global_allocator]
static ALLOC: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

fn main() -> Result<()> {
	color_eyre::install()?;
	let args = CliArgs::parse();
	match args.command {
		CliSubcommands::Scrub(args) => scrub::scrub(args),
		CliSubcommands::Test(_args) => todo!(),
	}
}
