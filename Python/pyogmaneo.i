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
%module ogmaneo

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

// Handle STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%include "typemaps.i"

%typemap(in) std::unordered_map<std::string, std::string>* (std::unordered_map<std::string, std::string> temp) {
  PyObject *key, *value;
  Py_ssize_t pos = 0;

  $1 = &temp;

  temp = std::unordered_map<std::string, std::string>();
  while (PyDict_Next($input, &pos, &key, &value)) {
    (*$1)[PyString_AsString(key)] = std::string(PyString_AsString(value));
  }
}

%typemap(argout) std::unordered_map<std::string, std::string>* {
 $result = PyDict_New();
 for(const auto& n : *$1) {
   PyDict_SetItemString($result, n.first.c_str(), PyString_FromString(n.second.c_str()));
 }
}

%include "system/SharedLib.h"
%include "system/ComputeSystem.h"
%include "system/ComputeProgram.h"
%include "neo/Hierarchy.h"
%include "neo/Architect.h"
