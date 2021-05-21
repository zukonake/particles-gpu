struct Part {
    float2 pos;
    float2 vel;
};

kernel void particle(const float t, const float d, global const struct Part* ins, global struct Part* outs) {
    size_t id = get_global_id(0);

    struct Part in = ins[id];

    float2 acc = (float2)(0.f, 0.f);
    for(size_t i = 0; i < PART_NUM; ++i) {
        if(i == id) continue;
        float2 d = ins[i].pos - in.pos;
        float l = sqrt(d.x * d.x + d.y * d.y);

        l = max(0.000001f, l);
        acc += (d / l);
    }

    in.vel += acc * t;
    in.pos += in.vel * 0.0000001f * t;
    in.vel *= d;

    outs[id] = in;
}
