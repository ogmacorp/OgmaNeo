// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

%begin %{
#include <cmath>
#include <iostream>
#include <unordered_map>
%}
%module csogmaneo

%{
#include "system/SharedLib.h"
#include "system/ComputeSystem.h"
#include "system/ComputeProgram.h"
#include "neo/Hierarchy.h"
#include "neo/Architect.h"
%}

%include "std_string.i"
%include "std_vector.i"

namespace std {
    %template(vectorf) vector<float>;
    %template(vectorvf) vector<ogmaneo::ValueField2D>;
};

%include "std_shared_ptr.i"
%shared_ptr(ogmaneo::Resources)
%shared_ptr(ogmaneo::ComputeSystem)
%shared_ptr(ogmaneo::ComputeProgram)
%shared_ptr(ogmaneo::Hierarchy)

%include "system/SharedLib.h"
%include "system/ComputeSystem.h"
%include "system/ComputeProgram.h"
%include "neo/Hierarchy.h"
%include "neo/Architect.h"
