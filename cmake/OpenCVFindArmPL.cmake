if(NOT AARCH64 AND NOT ARM64 AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    message(STATUS "ARM Performance Libraries: Not on ARM64 platform, skipping")
    return()
endif()

set(ARMPL_ROOT_DIR "" CACHE PATH "Path to ARM Performance Libraries root directory")

if(NOT ARMPL_ROOT_DIR)
    if(WIN32)
        file(GLOB ARMPL_BASE_PATHS 
            "C:/Program Files/Arm Performance Libraries/armpl*"
            "C:/Program Files/Arm/armpl*"
            "$ENV{ProgramFiles}/Arm Performance Libraries/armpl*"
            "$ENV{ProgramFiles}/Arm/armpl*"
        )
        if(ARMPL_BASE_PATHS)
            list(SORT ARMPL_BASE_PATHS ORDER DESCENDING)
            list(GET ARMPL_BASE_PATHS 0 ARMPL_ROOT_DIR)
            set(ARMPL_ROOT_DIR "${ARMPL_ROOT_DIR}" CACHE PATH "Path to ARM Performance Libraries root directory" FORCE)
        endif()
    endif()
endif()

if(NOT ARMPL_ROOT_DIR OR NOT EXISTS "${ARMPL_ROOT_DIR}")
    message(STATUS "ARM Performance Libraries: Root directory not found. Set ARMPL_ROOT_DIR to enable.")
    return()
endif()

message(STATUS "ARM Performance Libraries: Searching in ${ARMPL_ROOT_DIR}")

set(ARMPL_INCLUDE_DIR "${ARMPL_ROOT_DIR}/include")
if(NOT EXISTS "${ARMPL_INCLUDE_DIR}")
    message(STATUS "ARM Performance Libraries: include directory not found at ${ARMPL_INCLUDE_DIR}")
    return()
endif()

if(NOT EXISTS "${ARMPL_INCLUDE_DIR}/armpl.h")
    message(STATUS "ARM Performance Libraries: armpl.h not found in ${ARMPL_INCLUDE_DIR}")
    return()
endif()

set(ARMPL_LIBRARY_DIR "${ARMPL_ROOT_DIR}/lib")
if(NOT EXISTS "${ARMPL_LIBRARY_DIR}")
    message(STATUS "ARM Performance Libraries: library directory not found at ${ARMPL_LIBRARY_DIR}")
    return()
endif()

set(ARMPL_LIB_CANDIDATES
    "armpl_lp64_mp"
    "armpl_lp64"
    "armpl_ilp64_mp"
    "armpl_ilp64"
)

set(ARMPL_LIB_FOUND FALSE)
foreach(lib_candidate ${ARMPL_LIB_CANDIDATES})
    if(WIN32)
        set(ARMPL_LIB_FILE "${ARMPL_LIBRARY_DIR}/${lib_candidate}.lib")
    else()
        set(ARMPL_LIB_FILE "${ARMPL_LIBRARY_DIR}/lib${lib_candidate}.a")
    endif()
    
    if(EXISTS "${ARMPL_LIB_FILE}")
        set(ARMPL_LIB_NAME ${lib_candidate})
        set(ARMPL_LIB_FOUND TRUE)
        message(STATUS "ARM Performance Libraries: Found library variant ${lib_candidate}")
        break()
    endif()
endforeach()

if(NOT ARMPL_LIB_FOUND)
    message(STATUS "ARM Performance Libraries: No compatible library found in ${ARMPL_LIBRARY_DIR}")
    message(STATUS "    Searched for: armpl_lp64_mp.lib, armpl_lp64.lib, armpl_ilp64_mp.lib, armpl_ilp64.lib")
    return()
endif()

string(REGEX MATCH "armpl[_-]([0-9]+\\.[0-9]+\\.[0-9]+)" ARMPL_VERSION_MATCH "${ARMPL_ROOT_DIR}")
if(ARMPL_VERSION_MATCH)
    string(REGEX REPLACE "armpl[_-]" "" ARMPL_VERSION_STR "${ARMPL_VERSION_MATCH}")
else()
    set(ARMPL_VERSION_STR "unknown")
endif()

if(NOT TARGET armpl)
    add_library(armpl UNKNOWN IMPORTED)
    set_target_properties(armpl PROPERTIES
        IMPORTED_LOCATION "${ARMPL_LIB_FILE}"
        INTERFACE_INCLUDE_DIRECTORIES "${ARMPL_INCLUDE_DIR}"
    )
endif()

# For _mp variant, we need OpenMP
set(ARMPL_NEEDS_OPENMP FALSE)
if(ARMPL_LIB_NAME MATCHES "_mp$")
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(ARMPL_NEEDS_OPENMP TRUE)
        # Link OpenMP to the armpl target
        set_target_properties(armpl PROPERTIES
            INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX
        )
        message(STATUS "ARM Performance Libraries: OpenMP multi-threading enabled")
    else()
        message(FATAL_ERROR "ARM Performance Libraries: _mp variant requires OpenMP but it was not found")
    endif()
endif()

# Set variables in CACHE so they persist
if(ARMPL_NEEDS_OPENMP)
    set(ARMPL_LIBRARIES armpl OpenMP::OpenMP_CXX CACHE INTERNAL "ArmPL libraries")
else()
    set(ARMPL_LIBRARIES armpl CACHE INTERNAL "ArmPL libraries")
endif()
set(ARMPL_INCLUDE_DIRS "${ARMPL_INCLUDE_DIR}" CACHE INTERNAL "ArmPL include directories")
set(HAVE_ARMPL TRUE CACHE BOOL "ArmPL found and enabled" FORCE)
set(ARMPL_VERSION_STR "${ARMPL_VERSION_STR}" CACHE INTERNAL "ArmPL version")
set(ARMPL_LIB_NAME "${ARMPL_LIB_NAME}" CACHE INTERNAL "ArmPL library variant")

message(STATUS "ARM Performance Libraries: ENABLED")
message(STATUS "    Version:  ${ARMPL_VERSION_STR}")
message(STATUS "    Include:  ${ARMPL_INCLUDE_DIRS}")
message(STATUS "    Library:  ${ARMPL_LIB_FILE}")
message(STATUS "    Variant:  ${ARMPL_LIB_NAME}")