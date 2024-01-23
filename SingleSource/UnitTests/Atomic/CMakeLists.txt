# These tests output numerical values to improve debuggability.
# Setting FP_TOLERANCE causes the test checker to use fpcmp to compare
#   test and expected output.
# Because the nonatomic output may vary wildly even in a passing run,
#   we set FP_TOLERANCE high so the test will never fail due to
#   differing numerical results.
# The tests each have their own correctness checking and will fail
#   properly if something is actually wrong.
set(FP_TOLERANCE 1000000)

# Link the Clang built libatomic. 
execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libclang_rt.atomic.so
                OUTPUT_VARIABLE _path_to_libatomic
                OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(_libatomic_dir ${_path_to_libatomic} DIRECTORY)
add_link_options("LINKER:${_path_to_libatomic},-rpath=${_libatomic_dir}")

llvm_singlesource()
