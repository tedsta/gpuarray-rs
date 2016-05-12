use opencl;

use kernels::Kernels;

pub struct Context {
    pub device: opencl::hl::Device,
    pub ctx: opencl::hl::Context,
    pub queue: opencl::hl::CommandQueue,
    pub program: opencl::hl::Program,
    kernels: Kernels,
}

impl Context {
    pub fn new() -> Context {
        let program_src = format!("{}\n{}", include_str!("cl/main.cl"), include_str!("cl/slice_ops.cl"));

        let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

        println!("Using OpenCL Device: {}", device.name());

        let program = ctx.create_program_from_source(&program_src);
        program.build(&device).ok().expect("Couldn't build program.");

        // Create and store all of the kernels
        let kernels = Kernels::new(&program);

        Context {
            device: device,
            ctx: ctx,
            queue: queue,
            program: program,
            kernels: kernels,
        }
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }
}
