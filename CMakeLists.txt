# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.16)
# 项目信息 表示在项目Demo1中要用到CUDA和C++两种语言
project(gemm CUDA CXX)
#设置CUDA标准版本
set(CMAKE_CUDA_STANDARD 11)

#查找当前目录下所有的源文件，并将名称保存到DIR_SRCS变量中
aux_source_directory(. SOURCE_FILES)

# 导入CUDA包
find_package(CUDA)

#遍历每个源文件并生成对应的可执行程序
foreach(SOURCE_FILE ${SOURCE_FILES})
        #提取文件名作为可执行程序的名称，并添加.out扩展名
        get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)
        set (EXECUTABLE_NAME "${EXECUTABLE_NAME}.out")
        #使用显式的可执行程序名称，并将源文件添加到可执行程序
        add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})  
        #链接库
        target_link_libraries( ${EXECUTABLE_NAME} ${CUDA_cublas_LIBRARY} )  
        #设置CUDA可分离编译
        set_target_properties(
                ${EXECUTABLE_NAME}
                PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON)    
endforeach()



