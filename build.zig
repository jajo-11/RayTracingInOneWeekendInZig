const Builder = @import("std").build.Builder;

pub fn build(b: *Builder) void {
    const ray_tracing_exe = b.addExecutable("ray_tracing", "src/main.zig");
    ray_tracing_exe.setBuildMode(b.standardReleaseOptions());
    ray_tracing_exe.install();
}