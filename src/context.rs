use opencl;

pub struct Context {
    pub device: opencl::hl::Device,
    pub ctx: opencl::hl::Context,
    pub queue: opencl::hl::CommandQueue,
    pub program: opencl::hl::Program,
}

impl Context {
    pub fn new() -> Context {
        let ker = include_str!("matrix_ops.ocl");

        let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

        println!("Using OpenCL Device: {}", device.name());

        let program = ctx.create_program_from_source(ker);
        program.build(&device).ok().expect("Couldn't build program.");

        Context {
            device: device,
            ctx: ctx,
            queue: queue,
            program: program,
        }
    }
}
