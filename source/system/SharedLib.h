// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
    #define OGMA_HELPER_DLL_IMPORT __declspec(dllimport)
    #define OGMA_HELPER_DLL_EXPORT __declspec(dllexport)
    #define OGMA_HELPER_DLL_LOCAL
#else
    #if __GNUC__ >= 4
        #define OGMA_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
        #define OGMA_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
        #define OGMA_HELPER_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
    #else
        #define OGMA_HELPER_DLL_IMPORT
        #define OGMA_HELPER_DLL_EXPORT
        #define OGMA_HELPER_DLL_LOCAL
    #endif
#endif

// Now we use the generic helper definitions above to define OGMA_API and OGMA_LOCAL.
// OGMA_API is used for the public API symbols. It either DLL imports or DLL exports
// (or does nothing for static build) OGMA_LOCAL is used for non-api symbols.

#ifdef OGMA_DLL // defined if OGMA is compiled as a DLL
    #ifdef OgmaNeo_EXPORTS // defined if we are building the OGMA DLL (instead of using it)
        #define OGMA_API OGMA_HELPER_DLL_EXPORT
    #else
        #define OGMA_API OGMA_HELPER_DLL_IMPORT
    #endif // OGMA_DLL_EXPORTS
    #define OGMA_LOCAL OGMA_HELPER_DLL_LOCAL
#else // OGMA_DLL is not defined: this means OGMA is a static lib.
    #define OGMA_API
    #define OGMA_LOCAL
#endif // OGMA_DLL
