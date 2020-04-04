const std = @import("std");
const ArrayList = std.ArrayList;
const math = std.math;

const Color = packed struct {
    r: u8,
    g: u8,
    b: u8,
};

const Vec3 = struct {
    x: f64 = 0,
    y: f64 = 0,
    z: f64 = 0,

    pub fn one() Vec3 {
        return Vec3 {.x = 1, .y = 1, .z = 1};
    }

    pub fn random(rand: *std.rand.Random) Vec3 {
        return Vec3 {.x = rand.float(f64), .y = rand.float(f64), .z = rand.float(f64)};
    }

    pub fn add(self: Vec3, other: Vec3) Vec3 {
        return Vec3 {.x = self.x + other.x, .y = self.y + other.y, .z = self.z + other.z};
    }

    pub fn sub(self: Vec3, other: Vec3) Vec3 {
        return Vec3 {.x = self.x - other.x, .y = self.y - other.y, .z = self.z - other.z};
    }

    pub fn mul(self: Vec3, scalar: f64) Vec3 {
        return Vec3 {.x = scalar * self.x, .y = scalar * self.y, .z = scalar * self.z};
    }

    pub fn mul_per_elem(self: Vec3, v: Vec3) Vec3 {
        return Vec3 {.x = self.x * v.x, .y = self.y * v.y, .z = self.z * v.z};
    }

    pub fn div(self: Vec3, scalar: f64) Vec3 {
        return Vec3 {.x = self.x / scalar, .y = self.y / scalar, .z = self.z / scalar};
    }

    pub fn len_sqr(self: Vec3) f64 {
        return self.dot(self);
    }

    pub fn len(self: Vec3) f64 {
        return @sqrt(len_sqr(self));
    }

    pub fn dot(self: Vec3, other: Vec3) f64 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    pub fn cross(self: Vec3, other: Vec3) Vec3 {
        return Vec3 {.x = self.y * other.z - self.z * other.y,
                     .y = self.z * other.x - self.x * other.z,
                     .z = self.x * other.y - self.y * other.x};
    }

    pub fn unit_vec(self: Vec3) Vec3 {
        return self.div(self.len());
    }

    pub fn to_Color(self: Vec3, samples_per_pixel: u32) Color {
        const v: Vec3 = self.div(@intToFloat(f64, samples_per_pixel));
        const r = @sqrt(v.x);
        const g = @sqrt(v.y);
        const b = @sqrt(v.z);
        return Color {
                .r = @floatToInt(u8, 255.0 * clamp(r, 0.0, 1.0)),
                .g = @floatToInt(u8, 255.0 * clamp(g, 0.0, 1.0)),
                .b = @floatToInt(u8, 255.0 * clamp(b, 0.0, 1.0))
            };
    }

    pub fn random_in_unit_sphere(rand: *std.rand.Random) Vec3 {
        while (true) {
            const v = Vec3.random(rand).mul(2).sub(Vec3.one());
            if (v.len_sqr() < 1) {
                return v;
            }
        }
    }

    pub fn random_in_unit_disk(rand: *std.rand.Random) Vec3 {
        while (true) {
            const p = Vec3 {.x = rand.float(f64), .y = rand.float(f64)};
            if (p.len_sqr() < 1) {
                return p;
            }
        }
    }

    pub fn random_unit_vector(rand: *std.rand.Random) Vec3 {
        const a = rand.float(f64)*2*math.pi;
        const z = rand.float(f64)*2-1;
        const r = @sqrt(1 - z*z);
        return Vec3 {.x = r*@cos(a), .y = r*@sin(a), .z = z};
    }

    pub fn reflect(self: Vec3, n: Vec3) Vec3 {
        return self.sub(n.mul(2*self.dot(n)));
    }

    pub fn refract(self: Vec3, normal: Vec3, refract_quotient: f64) Vec3 {
        const cos_theta = -self.dot(normal);
        const r_para = self.add(normal.mul(cos_theta)).mul(refract_quotient);
        const r_perp = normal.mul(-1 * @sqrt(1.0 - r_para.len_sqr()));
        return r_para.add(r_perp);
    }
};

fn clamp(x: f64, min: f64, max: f64) f64 {
    if (x < min) { return min; }
    if (x > max) { return max; }
    return x;
}

fn deg_to_rad(deg: f64) f64 {
    return deg * math.pi / 180;
}

fn schlick_aprox(cos: f64, refract_index: f64) f64 {
    var r0 = (1 - refract_index) / (1 + refract_index);
    r0 = r0 * r0;
    return r0 + (1-r0)*math.pow(f64, (1 - cos), 5);
}

const HittableContainer = struct {
    spheres: ArrayList(Sphere),

    pub fn init(allocator: *std.mem.Allocator) HittableContainer {
        return HittableContainer {
            .spheres = ArrayList(Sphere).init(allocator),
        };
    }

    pub fn deinit(self: HittableContainer) void {
        self.spheres.deinit();
    }

    pub fn append(self: *HittableContainer, foo: var) !void {
        const T = @TypeOf(foo);
        switch (@typeName(T)) {
            "Sphere" => {
                try self.spheres.append(foo);
            },
            else => {
                @compileError("'" ++ @typeName(T) ++ "' is not a hittable Object and therefore can not be appended to the Container.");
            }
        }
    }

    pub fn hit(self: *HittableContainer, r: Ray, t_min: f64, t_max: f64) ?HitRecord {
        var rv: ?HitRecord = null;
        var closest = t_max;

        for (self.spheres.span()) |obj| {
            if (obj.hit(r, t_min, closest)) |rec| {
                rv = rec;
                closest = rec.t;
            }
        }

        return rv;
    }
};

const Ray = struct {
    pos: Vec3,
    dir: Vec3,

    pub fn at(self: Ray, t: f64) Vec3 {
        return self.pos.add(self.dir.mul(t));
    }
};

const attenuation_scattered = struct {
    attenuation: Vec3,
    scattered: Ray
};

const Material = struct {
    scatterFn: fn (mat: *Material, r: Ray, rec: HitRecord, rand: *std.rand.Random) ?attenuation_scattered,
};

const LambertianMaterial = struct {
    albedo: Vec3,
    material: Material = Material {.scatterFn = scatter},

    fn scatter(mat: *Material, r: Ray, rec: HitRecord, rand: *std.rand.Random) ?attenuation_scattered {
        const self = @fieldParentPtr(LambertianMaterial, "material", mat);

        const scatter_dir = rec.normal.add(Vec3.random_unit_vector(rand));
        const scattered = Ray {.pos = rec.p, .dir = scatter_dir};
        return attenuation_scattered {.attenuation = self.albedo, .scattered = scattered};
    }
};

const MetalMaterial = struct {
    albedo: Vec3,
    fuzz: f64, // <= 1
    material: Material = Material {.scatterFn = scatter},

    fn scatter(mat: *Material, r: Ray, rec: HitRecord, rand: *std.rand.Random) ?attenuation_scattered {
        const self = @fieldParentPtr(MetalMaterial, "material", mat);

        const reflected = r.dir.unit_vec().reflect(rec.normal);
        const scattered = Ray {.pos = rec.p,
            .dir = reflected.add(Vec3.random_in_unit_sphere(rand).mul(self.fuzz))};
        if (scattered.dir.dot(rec.normal) > 0) {
            return attenuation_scattered {.attenuation = self.albedo, .scattered = scattered};        
        } else {
            return null;
        }
    }
};

const DielectricMaterial = struct {
    refract_index: f64,
    material: Material = Material {.scatterFn = scatter},

    fn scatter(mat: *Material, r: Ray, rec: HitRecord, rand: *std.rand.Random) ?attenuation_scattered {
        const self = @fieldParentPtr(DielectricMaterial, "material", mat);

        const refract_quotient = if (rec.front_face) 1.0 / self.refract_index else self.refract_index;

        const dir = r.dir.unit_vec();
        const cos_theta = math.min(-dir.dot(rec.normal), 1.0);
        const sin_theta = @sqrt(1.0 - cos_theta*cos_theta);

        if (refract_quotient * sin_theta > 1.0 or schlick_aprox(cos_theta, refract_quotient) > rand.float(f64)) {
            const reflected = dir.reflect(rec.normal);
            return attenuation_scattered {.attenuation = Vec3.one(), .scattered = Ray {.pos = rec.p, .dir = reflected}};
        }

        const refracted = dir.refract(rec.normal, refract_quotient);
        return attenuation_scattered {.attenuation = Vec3.one(), .scattered = Ray {.pos = rec.p, .dir = refracted}};
    }
};

const HitRecord = struct {
    p: Vec3,
    normal: Vec3,
    t: f64,
    front_face: bool,
    mat: *Material,

    pub fn set_face_normal(self: *HitRecord, r: Ray, outward_normal: Vec3) void {
        self.front_face = r.dir.dot(outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else outward_normal.mul(-1);
    }
};

const Sphere = struct {
    center: Vec3,
    radius: f64,
    mat: *Material,

    pub fn hit(self: Sphere, r: Ray, t_min: f64, t_max: f64) ?HitRecord {
        const oc = r.pos.add(self.center.mul(-1.0));
        const a = r.dir.len_sqr();
        const half_b = oc.dot(r.dir);
        const c = oc.len_sqr() - self.radius * self.radius;
        const discriminant = half_b*half_b -a*c;
        if (discriminant > 0) {
            const root = @sqrt(discriminant);
            var temp = -(half_b + root)/a;
            if (temp < t_max and temp > t_min) {
                const p = r.at(temp);
                var rv = HitRecord {
                    .p = p,
                    .t = temp,
                    .mat = self.mat,
                    .normal = undefined,
                    .front_face = undefined,
                };
                const outward_normal = p.add(self.center.mul(-1)).div(self.radius);
                rv.set_face_normal(r, outward_normal);
                return rv;
            }
            temp = (-half_b + root)/a;
            if (temp < t_max and temp > t_min) {
                const p = r.at(temp);
                var rv = HitRecord {
                    .p = p,
                    .t = temp,
                    .mat = self.mat,
                    .normal = undefined,
                    .front_face = undefined
                };
                const outward_normal = p.add(self.center.mul(-1)).div(self.radius);
                rv.set_face_normal(r, outward_normal);
                return rv;
            }
        }
        return null;
    }
};

const Camera = struct {
    lower_left_corner: Vec3 = Vec3 {.x = -2.0, .y = -1.0, .z = -1.0},
    horizontal: Vec3 = Vec3 {.x = 4.0, .y = 0.0, .z = 0.0},
    vertical: Vec3 = Vec3 {.x = 0.0, .y = 2.0, .z = 0.0},
    u: Vec3 = Vec3 {},
    v: Vec3 = Vec3 {},
    origin: Vec3 = Vec3 {},
    lens_radius: f64,

    pub fn init(pos: Vec3, lookat: Vec3, vup: Vec3, vfov: f64, aspect: f64, aperture: f64, focus_dist: f64) Camera {
        var rv =  Camera {.origin = pos,
                          .lens_radius = aperture / 2};

        const theta = deg_to_rad(vfov);
        const half_height = math.tan(theta/2);
        const half_width = aspect * half_height;

        const w = pos.sub(lookat).unit_vec();
        rv.u = vup.cross(w).unit_vec();
        rv.v = w.cross(rv.u);

        rv.lower_left_corner = pos.sub(rv.u.mul(half_width*focus_dist))
                                  .sub(rv.v.mul(half_height*focus_dist))
                                  .sub(w.mul(focus_dist));
        rv.horizontal = rv.u.mul(2*half_width*focus_dist);
        rv.vertical = rv.v.mul(2*half_height*focus_dist);

        return rv;
    }

    pub fn getRay(self: Camera, u: f64, v: f64, rand: *std.rand.Random) Ray {
        const rd = Vec3.random_in_unit_disk(rand).mul(self.lens_radius);
        const offset = self.u.mul(rd.x).add(self.v.mul(rd.y));

        return Ray {
            .pos = self.origin.add(offset),
            .dir = self.lower_left_corner
                       .add(self.horizontal.mul(u))
                       .add(self.vertical.mul(v))
                       .sub(self.origin)
                       .sub(offset)
        };
    }
};

const ImageInfo = struct {
    image_width: u32 = 1920,
    image_height: u32 = 1080,
    samples_per_pixel: u32 = 100,
    max_depth: u32 = 50,
};

fn ray_color(r: Ray, world: *HittableContainer, depth: u32, rand: *std.rand.Random) Vec3 {
    if (depth <= 0) {
        return Vec3 {};
    }
    if (world.hit(r, 0.001, math.inf_f64)) |rec| {
        if (rec.mat.scatterFn(rec.mat, r, rec, rand)) |pkg| {
            return ray_color(pkg.scattered, world, depth - 1, rand).mul_per_elem(pkg.attenuation);
        }
        return Vec3 {};
    } else {
        const t = 0.5*(r.dir.unit_vec().y +  1.0);
        return Vec3 {.x = 1 - t * 0.5, .y = 1 - t * 0.3, .z = 1};
    }
}


fn RenderContext(width: u32) type {
    return struct {
        pixels: [][width]Color,
        world: *HittableContainer,
        cam: *const Camera,
        j_start: usize,
        info: ImageInfo,
        seed: u64,

        pub fn render_segment(context: @This()) void {
            var prng = std.rand.DefaultPrng.init(context.seed);
            var rand = &prng.random;
            var j: u32 = 0;
            while (j < context.pixels.len) : (j += 1) {
                var i: u32 = 0;
                while (i < context.info.image_width) : (i += 1) {
                    var col = Vec3 {};
                    var s: u32 = 0;
                    while (s < context.info.samples_per_pixel) :  (s += 1) {
                        const u = (@intToFloat(f64, i) + rand.float(f64))/ @intToFloat(f64, context.info.image_width);
                        const v = (@intToFloat(f64, context.info.image_height - j - context.j_start)
                                + rand.float(f64)) / @intToFloat(f64, context.info.image_height);
                        const r = context.cam.getRay(u, v, rand);
                        col = col.add(ray_color(r, context.world, context.info.max_depth, rand));
                    }
                    context.pixels[j][i] = col.to_Color(context.info.samples_per_pixel);
                }
            }
        }
    };
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = &arena.allocator;

    var prng: std.rand.Xoroshiro128 = undefined;
    {
        var buf: [8]u8 = undefined;
        try std.crypto.randomBytes(buf[0..]);
        const seed = std.mem.readIntLittle(u64, buf[0..8]);
        prng = std.rand.DefaultPrng.init(seed);
    }
    
    const info = ImageInfo {};

    const image_buff = try allocator.create([info.image_height][info.image_width]Color);

    var glass = DielectricMaterial {.refract_index = 1.5};
    var LambertianList = ArrayList(LambertianMaterial).init(allocator);
    var MetalList = ArrayList(MetalMaterial).init(allocator);

    var world = HittableContainer.init(allocator);

    var ground_mat = LambertianMaterial {.albedo = Vec3 {.x = 0.5, .y = 0.5, .z = 0.5}};
    try world.append(Sphere {.center = Vec3 {.x = 0, .y = -1000, .z = 0}, .radius = 1000, .mat = &ground_mat.material});

    var a: i32 = -11;
    while (a < 11) : (a += 1) {
        var b: i32 = -11;
        while (b < 11) : (b += 1) {
            const choose_mat = prng.random.float(f64);
            const center = Vec3 {.x = @intToFloat(f64, a) + 0.9 * prng.random.float(f64), .y = 0.2,
                                 .z = @intToFloat(f64, b) + 0.9 * prng.random.float(f64)};
            if (center.sub(Vec3 {.x = 4, .y = 0.2}).len() > 0.9) {
                if (choose_mat < 0.8) {
                    try LambertianList.append( LambertianMaterial
                        {.albedo = Vec3.random(&prng.random).mul_per_elem(Vec3.random(&prng.random))});
                    try world.append(Sphere {.center = center, .radius = 0.2,
                        .mat = &LambertianList.span()[LambertianList.len - 1].material});
                } else if (choose_mat < 0.95) {
                    try MetalList.append( MetalMaterial {
                        .albedo = Vec3 {.x = prng.random.float(f64)*0.5+0.5,
                                         .z = prng.random.float(f64)*0.5+0.5,
                                         .y = prng.random.float(f64)*0.5+0.5},
                        .fuzz = prng.random.float(f64)*0.5+0.5});
                    try world.append(Sphere {.center = center, .radius = 0.2,
                        .mat = &MetalList.span()[MetalList.len - 1].material});
                } else {
                    try world.append(Sphere {.center = center, .radius = 0.2, .mat = &glass.material});
                }
            }
        }
    }

    try world.append(Sphere {.center = Vec3 {.y = 1}, .radius = 1.0, .mat = &glass.material});
    try LambertianList.append( LambertianMaterial {.albedo = Vec3 {.x = 0.4, .y = 0.2, .z = 0.1}});
    try world.append(Sphere {.center = Vec3 {.x = -4, .y = 1}, .radius = 1.0,
                         .mat = &LambertianList.span()[LambertianList.len - 1].material});
    try MetalList.append(MetalMaterial {.albedo = Vec3 {.x = 0.4, .y = 0.2, .z = 0.1},
        .fuzz = 0.0});
    try world.append(Sphere {.center = Vec3 {.x = 4, .y = 1},
        .radius = 1.0, .mat = &MetalList.span()[MetalList.len - 1].material});

    const lookfrom = Vec3 {.x = 13, .y = 2, .z = 3};
    const lookat = Vec3 {};
    const cam = Camera.init(lookfrom, lookat, Vec3 {.y = 1}, 20,
                            @intToFloat(f64, info.image_width) / @intToFloat(f64, info.image_height),
                            0.1, 10);

    const cpu_count = try std.Thread.cpuCount();
    const batch_size = info.image_height / cpu_count;
    var threads = ArrayList(*std.Thread).init(allocator);
    defer threads.deinit();

    const RContext = RenderContext(info.image_width);

    var index: usize = 0;
    while (index < cpu_count * batch_size) : (index += batch_size) {
        try threads.append(try std.Thread.spawn(
            RContext {
                .pixels = image_buff[index..(index + batch_size)],
                .world = &world,
                .cam = &cam,
                .j_start = index,
                .info = info,
                .seed = prng.random.int(u64)
            }, RContext.render_segment
        ));
    }
    if (info.image_height % cpu_count != 0) {
        try threads.append(try std.Thread.spawn(
            RContext {
                .pixels = image_buff[index..],
                .world = &world,
                .cam = &cam,
                .j_start = index,
                .info = info,
                .seed = prng.random.int(u64)
            }, RContext.render_segment
        ));
    }

    for (threads.span()) |thread| {
        thread.wait();
    }

    const current_dir = std.fs.cwd();
    var out_file = try current_dir.createFile("out.tiff", .{.read = true, .truncate = false});
    defer out_file.close();

    // Write Tiff header (See: https://www.adobe.io/content/dam/udp/en/open/standards/tiff/TIFF6.pdf)
    // TODO support systems with most significant endianess
    const tiff_header = [_]u8{
        // Header
        0x49, 0x49, // Byte Order (least significant)
        0x2A, 0x00, // Magic number
        0x08, 0x00, 0x00, 0x00, // 1st IFD offset
        // IFD
        0x0C, 0x00, // Field count
        0x00, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00 // Width as Long
        } ++ @bitCast([4]u8, info.image_width) ++ [_]u8{
        0x01, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00 // Image Length as Long
        } ++ @bitCast([4]u8, info.image_height) ++ [_]u8{
        0x02, 0x01, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x9E, 0x00, 0x00, 0x00, // BitsPerSample offset to values later
        0x03, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // Compression (off)
        0x06, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, // PhotometricInterpretation (2=RGB)
        0x11, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0xB4, 0x00, 0x00, 0x00, //StripOffset
        0x15, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, // SamplesPerPixel (3)
        0x16, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00
        } ++ @bitCast([4]u8, info.image_height) ++ [_]u8{//RowsPerStrip
        0x17, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00
        } ++ @bitCast([4]u8, info.image_height*info.image_width*3) ++ [_]u8{ //StripByteCounts
        0x1A, 0x01, 0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0xA4, 0x00, 0x00, 0x00, // XResolution offset to values
        0x1B, 0x01, 0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0xAC, 0x00, 0x00, 0x00, // YResolution offset to values
        0x28, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // ResolutionUnit (1=No absolute unit of measurement) 2->inch 3->cm
        0x00, 0x00, 0x00, 0x00, // Next IFD offset (none)
        0x08, 0x00, 0x08, 0x00, 0x08, 0x00, // BitsPerSample 8,8,8
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // XReselution
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // YReselution
        };

    const n_written = try out_file.write(tiff_header[0..]);
    const m_written = try out_file.write(std.mem.sliceAsBytes(image_buff[0..]));
    std.debug.warn("\nDone.\n", .{});
}
