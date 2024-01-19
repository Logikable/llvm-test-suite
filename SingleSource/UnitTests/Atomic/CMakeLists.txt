set(FP_TOLERANCE 1000000)
execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libclang_rt.atomic.so
                OUTPUT_VARIABLE _path_to_libatomic
                OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(_libatomic_dir ${_path_to_libatomic} DIRECTORY)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${_path_to_libatomic} -Wl,-rpath=${_libatomic_dir}")

llvm_singlesource()
