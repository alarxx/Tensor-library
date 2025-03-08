/*
    SPDX-License-Identifier: MPL-2.0
    --------------------------------
    This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
    If a copy of the MPL was not distributed with this file,
    You can obtain one at https://mozilla.org/MPL/2.0/.

    This file is part of the Tensor-library:
    https://github.com/alarxx/Tensor-library

    Provided “as is”, without warranty of any kind.

    Copyright © 2025 Alar Akilbekov. All rights reserved.
 */

// --- Logging ---
template <typename T>
concept __is_stream_supported = requires (T t){
    // SFINAE constrains
    std::declval<std::ostream&>() << t;
};

template <typename T>
requires __is_stream_supported<T>
void log(T t){
    #if DEBUG_LOG_TENSOR
        std::cout << t << std::endl;
    #endif
}

template <typename T, typename ... TArgs>
requires __is_stream_supported<T>
void log(T t, TArgs ... args){
    #if DEBUG_LOG_TENSOR
        std::cout << t;
        log(args...);
    #endif
}
// ------
