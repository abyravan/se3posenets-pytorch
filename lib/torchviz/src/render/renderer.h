#ifndef L2S_RENDERER_H_
#define L2S_RENDERER_H_

#include <pangolin/pangolin.h>
#include "util/type_list.h"

namespace l2s {

std::string compileDirectory();

// -=-=-=- render type stuff -=-=-=-
enum RenderType {
    RenderColor=0,
    RenderVertMap,
    RenderVertMapWMeshID,
    RenderDepth,
    RenderNormals
};

template <int I>
struct IntToType {
    enum { value = I };
};

template <typename RT>
struct RenderTypeTraits;

template <>
struct RenderTypeTraits<IntToType<RenderColor> > {
    static const char * vertShaderName() {
        static const char name[] = "color.vert";
        return name;
    }
    static const char * fragShaderName() {
        static const char name[] = "color.frag";
        return name;
    }
    static constexpr int channels = 3;
    static constexpr GLint texFormat = GL_RGB8;
    static constexpr int numVertAtrributes = 2;
    static const int * vertAttributeSizes() {
        static const int sizes[numVertAtrributes] = { 3, 3 };
        return sizes;
    }
    static const GLint * vertAttributeTypes() {
        static const GLint types[numVertAtrributes] = { GL_FLOAT, GL_FLOAT };
        return types;
    }
};

template <>
struct RenderTypeTraits<IntToType<RenderVertMap> > {
    static const char * vertShaderName() {
        static const char name[] = "vertmap.vert";
        return name;
    }
    static const char * fragShaderName() {
        static const char name[] = "vertmap.frag";
        return name;
    }
    static constexpr int channels = 3;
    static constexpr GLint texFormat = GL_RGB32F;
    static constexpr int numVertAtrributes = 2;
    static const int * vertAttributeSizes() {
        static const int sizes[numVertAtrributes] = { 3, 3 };
        return sizes;
    }
    static const GLint * vertAttributeTypes() {
        static const GLint types[numVertAtrributes] = { GL_FLOAT, GL_FLOAT };
        return types;
    }
};

template <>
struct RenderTypeTraits<IntToType<RenderVertMapWMeshID> > {
    static const char * vertShaderName() {
        static const char name[] = "vertmapwmeshid.vert";
        return name;
    }
    static const char * fragShaderName() {
        static const char name[] = "vertmapwmeshid.frag";
        return name;
    }
    static constexpr int channels = 4;
    static constexpr GLint texFormat = GL_RGBA32F;
    static constexpr int numVertAtrributes = 2;
    static const int * vertAttributeSizes() {
        static const int sizes[numVertAtrributes] = { 3, 4 };
        return sizes;
    }
    static const GLint * vertAttributeTypes() {
        static const GLint types[numVertAtrributes] = { GL_FLOAT, GL_FLOAT };
        return types;
    }
};

template <>
struct RenderTypeTraits<IntToType<RenderNormals> > {
    static const char * vertShaderName() {
        static const char name[] = "normals.vert";
        return name;
    }
    static const char * fragShaderName() {
        static const char name[] = "normals.frag";
        return name;
    }
    static constexpr int channels = 3;
    static constexpr GLint texFormat = GL_RGB32F;
    static constexpr int numVertAtrributes = 2;
    static const int * vertAttributeSizes() {
        static const int sizes[numVertAtrributes] = { 3, 3 };
        return sizes;
    }
    static const GLint * vertAttributeTypes() {
        static const GLint types[numVertAtrributes] = { GL_FLOAT, GL_FLOAT };
        return types;
    }
};

template <>
struct RenderTypeTraits<IntToType<RenderDepth> > {
    static const char * vertShaderName() {
        static const char name[] = "depth.vert";
        return name;
    }
    static const char * fragShaderName() {
        static const char name[] = "depth.frag";
        return name;
    }
    static constexpr int channels = 1;
    static constexpr GLint texFormat = GL_LUMINANCE32F_ARB;
    static constexpr int numVertAtrributes = 1;
    static const int * vertAttributeSizes() {
        static const int sizes[numVertAtrributes] = { 3 };
        return sizes;
    }
    static const GLint * vertAttributeTypes() {
        static const GLint types[numVertAtrributes] = { GL_FLOAT };
        return types;
    }
};

// -=-=-=- framebuffer type stuff -=-=-=-
template<GLint F=GL_RGB8UI>
struct FrameBufferType {
    enum { format = F };
};

template <typename RenderTypeList>
struct CreateFrameBufferTypeList;

template <typename T1>
struct CreateFrameBufferTypeList<TypeList<T1,NullType> > {
    typedef TypeList<FrameBufferType<RenderTypeTraits<T1>::texFormat>, NullType > result;
};

template <typename T1, typename T2>
struct CreateFrameBufferTypeList<TypeList<T1,T2> > {
    typedef TypeList<FrameBufferType<RenderTypeTraits<T1>::texFormat>, typename CreateFrameBufferTypeList<T2>::result > result;
};

// -=-=-=- framebuffer holder -=-=-=-
template <typename FBType>
class FrameBufferHolder {
public:

    explicit FrameBufferHolder(const int width, const int height) :
        colorBuffer_(width,height,FBType::format),
        renderBuffer_(width,height),
        frameBuffer_(colorBuffer_,renderBuffer_) { }

    pangolin::GlTexture & colorBuffer() { return colorBuffer_; }

    pangolin::GlRenderBuffer & renderBuffer() { return renderBuffer_; }

    pangolin::GlFramebuffer & frameBuffer() { return frameBuffer_; }

private:
    pangolin::GlTexture colorBuffer_;
    pangolin::GlRenderBuffer renderBuffer_;
    pangolin::GlFramebuffer frameBuffer_;
};

// -=-=-=- render progam holder -=-=-=-
template <typename RenderType>
class RenderProgramHolder {
public:

    RenderProgramHolder() {
        const std::string shaderDir(compileDirectory() + "/shaders/");
        const std::string vertPath = shaderDir + RenderTypeTraits<RenderType>::vertShaderName();
        const std::string fragPath = shaderDir + RenderTypeTraits<RenderType>::fragShaderName();
        //std::cout << "adding " << vertPath << std::endl;
        program_.AddShaderFromFile(pangolin::GlSlVertexShader, vertPath);
        //std::cout << "adding " << fragPath << std::endl;
        program_.AddShaderFromFile(pangolin::GlSlFragmentShader, fragPath);
        //std::cout << "linking" << std::endl;
        program_.Link();
        projectionMatrixHandle_ = program_.GetUniformHandle("projectionMatrix");
        modelviewMatrixHandle_ = program_.GetUniformHandle("modelViewMatrix");
    }

    inline pangolin::GlSlProgram & program() { return program_; }

    inline GLint projectionMatrixHandle() { return projectionMatrixHandle_; }

    inline GLint modelviewMatrixHandle() { return modelviewMatrixHandle_; }

private:
    pangolin::GlSlProgram program_;
    GLint projectionMatrixHandle_;
    GLint modelviewMatrixHandle_;
};

// -=-=-=- renderer -=-=-=-
template <typename RenderTypeList>
class Renderer {
public:

    explicit Renderer(const int renderWidth, const int renderHeight, const Eigen::Matrix4f & projectionMatrix)
        : renderWidth_(renderWidth),
          renderHeight_(renderHeight),
          projectionMatrix_(projectionMatrix),
          frameBuffers_(renderWidth,renderHeight) { }

    template <RenderType RT>
    void renderMeshes(const std::vector<std::vector<pangolin::GlBuffer *> > & vertexAttributeBuffers,
                      const std::vector<pangolin::GlBuffer> & indexBuffers);

    template <RenderType RT>
    pangolin::GlTexture & texture();

    void setProjectionMatrix(const Eigen::Matrix4f & m) { projectionMatrix_ = m; }

    void setModelViewMatrix(const Eigen::Matrix4f & m) { modelViewMatrix_ = m; }

private:

    // -=-=-=- typedefs -=-=-=-
    typedef typename CreateFrameBufferTypeList<RenderTypeList>::result FrameBufferTypeListWithDuplicates;
    typedef typename NoDuplicates<FrameBufferTypeListWithDuplicates>::result FrameBufferTypeList;

    typedef GenScatteredFBHierarchy<FrameBufferTypeList,FrameBufferHolder> FrameBuffers;
    typedef GenScatteredHierarchy<RenderTypeList,RenderProgramHolder> RenderPrograms;

    // -=-=-=- members -=-=-=-
    FrameBuffers frameBuffers_;
    RenderPrograms renderPrograms_;

    const int renderWidth_;
    const int renderHeight_;

    Eigen::Matrix4f projectionMatrix_;
    Eigen::Matrix4f modelViewMatrix_;

    // -=-=-=- methods -=-=-=-
    template <RenderType RT>
    inline pangolin::GlFramebuffer & frameBuffer();

    template <RenderType RT>
    inline void bindFrameBuffer() { frameBuffer<RT>().Bind(); }

    template <RenderType RT>
    inline void unbindFrameBuffer() { frameBuffer<RT>().Unbind(); }


    template <RenderType RT>
    inline pangolin::GlSlProgram & program();

    template <RenderType RT>
    inline void bindProgram() { program<RT>().Bind(); }

    template <RenderType RT>
    inline void unbindProgram() { program<RT>().Unbind(); }


    template <RenderType RT>
    inline GLint projectionMatrixHandle();

    template <RenderType RT>
    inline GLint modelViewMatrixHandle();
};

// -=-=-=-=-=-=-=-=-=-=-=- Implementations -=-=-=-=-=-=-=-=-=-=-=-
template <typename RenderTypeList>
template <RenderType RT>
inline pangolin::GlFramebuffer & Renderer<RenderTypeList>::frameBuffer() {
    pangolin::GlFramebuffer & fb = static_cast<FrameBufferHolder<
            FrameBufferType<
              RenderTypeTraits<IntToType<RT> >::texFormat
            >
          >&>(frameBuffers_).frameBuffer();
    return fb;
}

template <typename RenderTypeList>
template <RenderType RT>
inline pangolin::GlSlProgram & Renderer<RenderTypeList>::program() {
    pangolin::GlSlProgram & program = static_cast<RenderProgramHolder<IntToType<RT> >&>(renderPrograms_).program();
    return program;
}

template <typename RenderTypeList>
template <RenderType RT>
inline GLint Renderer<RenderTypeList>::modelViewMatrixHandle() {
    return static_cast<RenderProgramHolder<IntToType<RT> >&>(renderPrograms_).modelviewMatrixHandle();
}

template <typename RenderTypeList>
template <RenderType RT>
inline GLint Renderer<RenderTypeList>::projectionMatrixHandle() {
    return static_cast<RenderProgramHolder<IntToType<RT> >&>(renderPrograms_).projectionMatrixHandle();
}


template <typename RenderTypeList>
template <RenderType RT>
inline pangolin::GlTexture & Renderer<RenderTypeList>::texture() {
    pangolin::GlTexture & tex = static_cast<FrameBufferHolder<
            FrameBufferType<
              RenderTypeTraits<IntToType<RT> >::texFormat
            >
          >&>(frameBuffers_).colorBuffer();
    return tex;
}


template <typename RenderTypeList>
template <RenderType RT>
void Renderer<RenderTypeList>::renderMeshes(const std::vector<std::vector<pangolin::GlBuffer *> > & vertexAttributeBuffers,
                                            const std::vector<pangolin::GlBuffer> & indexBuffers) {

    const int nMeshes = vertexAttributeBuffers.size();
    assert(nMeshes == indexBuffers.size());

    // activate, scissor, and clear frame buffer
    bindFrameBuffer<RT>();
    //glClearColor(std::nanf(""),std::nanf(""),std::nanf(""),std::nanf(""));
    //glClearColor(1,1,1,1);
    glEnable(GL_SCISSOR_TEST);
    glViewport(0,0,renderWidth_,renderHeight_);
    glScissor(0,0,renderWidth_,renderHeight_);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    // bind shader and set matrices
    bindProgram<RT>();

    glUniformMatrix4fv(projectionMatrixHandle<RT>(), 1, GL_FALSE, projectionMatrix_.data());
    glUniformMatrix4fv(modelViewMatrixHandle<RT>(), 1, GL_FALSE, modelViewMatrix_.data());

    // set vertex attributes and draw elements
    for (int m = 0; m < nMeshes; ++m) {

        // make sure the number of vertex attribute buffers provided is correct
        const int nAttributeBuffers = vertexAttributeBuffers[m].size();
        assert(nAttributeBuffers == RenderTypeTraits<IntToType<RT> >::numVertAtrributes);

        for (GLuint i=0; i<RenderTypeTraits<IntToType<RT> >::numVertAtrributes; ++i) {
            const GLuint size = RenderTypeTraits<IntToType<RT> >::vertAttributeSizes()[i];
            const GLenum type = RenderTypeTraits<IntToType<RT> >::vertAttributeTypes()[i];

            // check buffer types
            assert(size == vertexAttributeBuffers[m][i]->count_per_element);
            assert(type == vertexAttributeBuffers[m][i]->datatype);

            // bind buffer and enable attribute
            vertexAttributeBuffers[m][i]->Bind();
            glEnableVertexAttribArray(i);
            glVertexAttribPointer(i, size, type, GL_FALSE, 0, 0);
        }

        indexBuffers[m].Bind();
        glDrawElements(GL_TRIANGLES, indexBuffers[m].num_elements, GL_UNSIGNED_INT, 0);

        indexBuffers[m].Unbind();

        for (uint i=0; i<nAttributeBuffers; ++i) {
            vertexAttributeBuffers[m][i]->Unbind();
        }
    }

    for (int i=0; i < RenderTypeTraits<IntToType<RT> >::numVertAtrributes; ++i) {
        glDisableVertexAttribArray(i);
    }

    // unbind
    unbindProgram<RT>();
    unbindFrameBuffer<RT>();

}

} // namespace l2s

#endif // L2S_RENDERER_H_
