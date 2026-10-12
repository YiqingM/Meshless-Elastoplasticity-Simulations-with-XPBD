cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)

# Local
set(SOURCE_FILES
    main.cpp)
    
add_executable(learnSPH_app ${SOURCE_FILES})
add_dependencies(learnSPH_app learnSPH)
target_link_libraries(learnSPH_app learnSPH)
add_executable(learnSPH_sand_collapse sand_collapse.cpp)
target_link_libraries(learnSPH_sand_collapse learnSPH)
add_executable(learnSPH_cantilever cantilever.cpp)
target_link_libraries(learnSPH_cantilever learnSPH)