/*
    SPDX-License-Identifier: MPL-2.0
    --------------------------------
    This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
    If a copy of the MPL was not distributed with this file,
    You can obtain one at https://mozilla.org/MPL/2.0/.

    Provided “as is”, without warranty of any kind.

    Copyright © 2026 Alar Akilbekov. All rights reserved.
*/

#pragma once
#ifndef _UTILS_
#define _UTILS_

#include <string>

namespace tensor {

template<typename T>
std::string array2string(int length, const T * array){
    std::string result = "[";
    for(int i = 0; i < length; ++i){
        result += std::to_string(array[i]);
        if (i != length - 1) result += ", ";
    }
    result += "]";
    return result;
}

};

#endif
