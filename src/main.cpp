#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <chrono>
#include <thread>
#include <math.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
//
#include <CL/cl.h>
#include "SimplexNoise.h"

typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float    f32;
typedef double   f64;

f32 constexpr tau = 6.28318530717958647692528676655900576839433879875021f;

struct v2 { f32 x, y; };

constexpr size_t part_num = 1 << 16;

GLFWwindow *win;

cl_context       context;
cl_program       program;
cl_kernel        kernel;
cl_command_queue command_queue;

cl_mem           cl_part_master;
cl_mem           cl_part[2];

GLuint           gl_program;
GLuint           vbo;
GLuint           vao;

struct Part {
    cl_float2 pos;
    cl_float2 vel;
};

Part* parts;

f32 randf() {
    return (f32)(rand() % (1 << 20)) / (f32)((1 << 20) - 1);
}

GLuint load_shader(GLenum type, char const* str) {
    FILE* f = fopen(str, "r");
    assert(f != nullptr);
    assert(fseek(f, 0, SEEK_END) == 0);
    long int len = ftell(f);
    assert(len != -1L);
    fseek(f, 0, SEEK_SET);
    char* buff = new char[len + 1];
    assert(fread(buff, len, 1, f) == 1);
    assert(fclose(f) == 0);
    buff[len] = '\0';

    GLuint id = glCreateShader(type);
    glShaderSource(id, 1, &buff, nullptr);
    glCompileShader(id);

    {   int success;
        glGetShaderiv(id, GL_COMPILE_STATUS, &success);
        if(!success) {
            static constexpr size_t OPENGL_LOG_SIZE = 512;
            char log[OPENGL_LOG_SIZE];
            glGetShaderInfoLog(id, OPENGL_LOG_SIZE, nullptr, log);
            fprintf(stderr, type == GL_VERTEX_SHADER ? "vert\n" : "frag\n");
            fprintf(stderr, "shader compilation error: \n%s", log);
        }
    }
    delete[] buff;
    return id;
}

GLuint load_program(char const* vert, char const* frag) {
    GLuint vert_id = load_shader(GL_VERTEX_SHADER  , vert);
    GLuint frag_id = load_shader(GL_FRAGMENT_SHADER, frag);

    GLuint id = glCreateProgram();
    glAttachShader(id, vert_id);
    glAttachShader(id, frag_id);
    glLinkProgram(id);
    glDeleteShader(vert_id);
    glDeleteShader(frag_id);

    {   int success;
        glGetProgramiv(id, GL_LINK_STATUS, &success);
        if(!success) {
            static constexpr size_t OPENGL_LOG_SIZE = 512;
            char log[OPENGL_LOG_SIZE];
            glGetProgramInfoLog(id, OPENGL_LOG_SIZE, nullptr, log);
            fprintf(stderr, "program linking error: \n%s", log);
        }
    }
    return id;

}


void init_gl()
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    win = glfwCreateWindow(1600, 900, "particles-gl", glfwGetPrimaryMonitor(), NULL);
    glfwMakeContextCurrent(win);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(win, GLFW_STICKY_MOUSE_BUTTONS, 1);
    glfwSwapInterval(0);

    if(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0)
    {
        fprintf(stderr, "failed to init glad\n");
        abort();
    }

    glViewport(0, 0, 1600, 900);

    gl_program = load_program("src/vert.glsl", "src/frag.glsl");

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Part), (void*)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Part), (void*)offsetof(Part, vel));
}

f32 fbm(f32 s, u32 octaves) {
    f32 val = 0.f;
    f32 w   = .5f;
    for(u32 i = 0; i < octaves; ++i) {
        val += SimplexNoise::noise(s) * w;
        s *= 2.f;
        w *= .5f;
    }
    return val / (1.f - w * 2.f);
}

#define check_cl_error(err) \
    if(err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error on line %d %d\n", __LINE__, err); \
        abort(); \
    } \

void init_cl()
{
    const char *program_path = "src/particle.cl";

    cl_uint platform_id_count = 0;
    cl_int err = CL_SUCCESS;
    clGetPlatformIDs(0, nullptr, &platform_id_count);

    if(platform_id_count == 0)
    {
        fprintf(stderr, "no opencl platforms\n");
        abort();
    }

    cl_platform_id* platform_ids = new cl_platform_id[platform_id_count]; 
    clGetPlatformIDs(platform_id_count, platform_ids, nullptr);

    cl_uint device_id_count = 0;
    clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, 0, nullptr,
        &device_id_count);

    if(device_id_count == 0)
    {
        fprintf(stderr, "no opencl devices\n");
        abort();
    }

    cl_device_id* device_ids = new cl_device_id[device_id_count]; 
    clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, device_id_count,
        device_ids, nullptr);

    const cl_context_properties context_properties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform_ids[0]),
        0, 0
    };

    context = clCreateContext(context_properties, device_id_count,
        device_ids, nullptr, nullptr, &err);
    check_cl_error(err);
    FILE* f = fopen(program_path, "r");
    if(f == nullptr)
    {
        fprintf(stderr, "failed to load opencl program\n");
        abort();
    }
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buff = new char[len];
    assert(fread(buff, 1, len, f) == len);
    fclose(f);
    
    size_t src_len[1]       = { len };
    char const *src_data[1] = { buff };

    program = clCreateProgramWithSource(context, 1, src_data, src_len, nullptr);

    {
        size_t sl = snprintf(nullptr, 0, "-DPART_NUM=%zu", part_num);
        char* b = new char[sl];
        snprintf(b, sl, "-DPART_NUM=%zu", part_num);
        clBuildProgram(program, device_id_count, device_ids,
                       b, nullptr, nullptr);
        delete[] b;
    }
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        char* log = new char[log_size];

        clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG,
                              log_size, log, nullptr);

        fprintf(stderr, "%.*s\n", (int)log_size, log);
        delete[] log;
    }

    kernel = clCreateKernel(program, "particle", &err);
    check_cl_error(err);

    command_queue = clCreateCommandQueueWithProperties(context, device_ids[0],
        nullptr, &err);
    check_cl_error(err);
    f32 seed = randf() * 1000.f;

    for(size_t i = 0; i < part_num; ++i) {
        v2 bob;
        float aa = tau * fbm((f32)i * 0.0001f + seed, 16);
        //float aa = randf() * tau;
        float a = aa + tau / 8.f;
        bob.x = cos(aa) * 1.f;
        bob.y = sin(aa) * 1.f;
        float l = randf();//fbm((f32)i * 0.0001f + seed, 16);
        //float l = fbm((f32)i * 0.0001f + seed, 16);
        //if(l < 0.1f) l = 0.1f;

        parts[i].pos = {bob.x * l, bob.y * l};
        float v = glm::length(glm::vec2(bob.x, bob.y)) * 0.001f;
        v2 dong;
        //dong.x = fbm((f32)i * 0.0001f + seed, 8);
        //dong.y = fbm((f32)(i + part_num) * 0.0001f + seed, 8);
        dong.x = cos(a);
        dong.y = sin(a);

        v2 vel = {dong.x, dong.y};
        //float len = sqrt(vel.x * vel.x + vel.y * vel.y);
        parts[i].vel = {(vel.x / len) * v, (vel.y / len) * v};
    }

    Part* parts_temp = new Part[part_num * 2];
    memcpy(parts_temp, parts, part_num * sizeof(Part));
    memcpy(parts_temp + part_num, parts, part_num * sizeof(Part));

    cl_part_master = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(Part) * part_num * 2, parts_temp, &err);
    check_cl_error(err);

    struct {
        size_t origin = 0;
        size_t sz     = sizeof(Part) * part_num;
    } region0;
    cl_part[0] = clCreateSubBuffer(cl_part_master, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &region0, &err);
    check_cl_error(err);

    struct {
        size_t origin = sizeof(Part) * part_num;
        size_t sz     = sizeof(Part) * part_num;
    } region1;
    cl_part[1] = clCreateSubBuffer(cl_part_master, CL_MEM_READ_WRITE,
        CL_BUFFER_CREATE_TYPE_REGION, &region1, &err);
    check_cl_error(err);

    delete[] platform_ids;
    delete[] device_ids;
    delete[] buff;
}

void check_gl_error() {
    GLenum error = glGetError();
    if(error != GL_NO_ERROR)
    {
        const char* str;
        switch(error)
        {
            case GL_INVALID_ENUM: str = "invalid enum"; break;
            case GL_INVALID_VALUE: str = "invalid value"; break;
            case GL_INVALID_OPERATION: str = "invalid operation"; break;
            case GL_OUT_OF_MEMORY: str = "out of memory"; break;
            default: str = "unknown"; break;
        }
        fprintf(stderr, "opengl error %s\n", str);
        abort();
    }
}

f32 time_scale = 0.01f;
f32 deacc      = 0.006f;

cl_mem* cast()
{
    cl_mem* part_in;
    cl_mem* part_out;
    static bool dong = false;
    part_in  = dong ? &cl_part[1] : &cl_part[0];
    part_out = dong ? &cl_part[0] : &cl_part[1];
    dong = not dong;

    cl_uint err;
    err = clSetKernelArg(kernel, 0, sizeof(f32), &time_scale);
    check_cl_error(err);
    f32 real_deacc = 1.f - deacc;
    err = clSetKernelArg(kernel, 1, sizeof(f32), &real_deacc);
    check_cl_error(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), part_in);
    check_cl_error(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), part_out);
    check_cl_error(err);

    constexpr size_t global_work_size[] = {part_num};
    constexpr size_t local_work_size[]  = {256};

    cl_event event;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1,
        nullptr, global_work_size, local_work_size, 0, nullptr, &event);
    check_cl_error(err);
    clWaitForEvents(1, &event);
    clFinish(command_queue);
    return part_out;
}

f32 scale = 1.f;

void scroll_cb(GLFWwindow* window, f64 d_x, f64 d_y) {
    d_y *= 0.1f;
    d_y += 1.f;
    if(scale + d_y >= 0.001f) {
        scale *= d_y;
    } else {
        scale = 0.001f;
    }
}

int main()
{
    parts = new Part[part_num];
    srand(time(NULL));
    init_gl();
    init_cl();
    glfwSetScrollCallback(win, scroll_cb);
    GLint loc = glGetUniformLocation(gl_program, "scale");
    GLint pos_loc = glGetUniformLocation(gl_program, "cpos");
    v2 pos = {0.f, 0.f};
    glClearColor(0, 0, 0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    while(not glfwWindowShouldClose(win))
    {
        auto before = std::chrono::steady_clock::now();
        glfwPollEvents();

        cl_mem* part_out = cast();

        cl_event event;
        clEnqueueReadBuffer(command_queue, *part_out, CL_TRUE,
            0, sizeof(Part) * part_num, parts, 0, nullptr, &event);
        clWaitForEvents(1, &event);
        clFinish(command_queue);
        if(glfwGetKey(win, GLFW_KEY_W)) pos.y += -1.f / scale;
        if(glfwGetKey(win, GLFW_KEY_S)) pos.y +=  1.f / scale;
        if(glfwGetKey(win, GLFW_KEY_A)) pos.x +=  1.f / scale;
        if(glfwGetKey(win, GLFW_KEY_D)) pos.x += -1.f / scale;
        if(glfwGetKey(win, GLFW_KEY_R)) time_scale *= 1.05f;
        if(glfwGetKey(win, GLFW_KEY_F)) time_scale /= 1.05f;
        if(glfwGetKey(win, GLFW_KEY_T)) deacc *= 1.05f;
        if(glfwGetKey(win, GLFW_KEY_G)) deacc /= 1.05f;
        time_scale = std::max(0.00000000000001f, time_scale);
        if(time_scale > 1.00000) time_scale = 1.00000;
        if(deacc < 0.0006) deacc = 0.0006;
        if(deacc > 0.5000) deacc = 0.5000;

        glUseProgram(gl_program);
        //glPointSize(1.f);
        glEnable(GL_BLEND);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);  
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_ALPHA);  

        //glClearColor(0, 0, 0, 0.5);
        //glClear(GL_COLOR_BUFFER_BIT);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Part) * part_num, parts, GL_DYNAMIC_DRAW);

        glBindVertexArray(vao);
        glUniform2f(pos_loc, pos.x * 0.01f, pos.y * 0.01f);
        glUniform1f(loc, scale);
        glDrawArrays(GL_POINTS, 0, part_num);
        
        glfwSwapBuffers(win);

        std::chrono::duration<double> t = std::chrono::steady_clock::now() - before;
        auto tick = std::chrono::duration<double>(1.0 / 60.0) - t;
        if(tick > std::chrono::duration<double>::zero()) {
            std::this_thread::sleep_for(tick);
        }
    }

    glfwTerminate();

    clReleaseMemObject(cl_part[0]);
    clReleaseMemObject(cl_part[1]);
    clReleaseMemObject(cl_part_master);
    clReleaseCommandQueue(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    delete[] parts;
    return 0;
}

