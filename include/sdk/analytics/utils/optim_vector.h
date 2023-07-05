/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_OPTIM_VECTOR_H
#define METAVISION_SDK_ANALYTICS_OPTIM_VECTOR_H

#include <assert.h>
#include <stdexcept>
#include <vector>

namespace Metavision {

/// @brief Class to avoid reallocating memory after clear() in case of a 2-dimensional vector.
///
/// The vector itself isn't cleared, but the end() iterator is moved back to begin().
/// Items of type @p T might contain vectors. The idea is to clear an existing Item and rewrite values on it, instead
/// of creating a new object and reallocating memory. Such @p T elements have only a default constructor.
///
/// To be used in the case one can only clear the vector or add an element at the end.
///
/// @tparam T Type of the elements, implementing a void clear() method
template<typename T>
class OptimVector {
public:
    using const_iterator_type = typename std::vector<T>::const_iterator;

    /// @brief Default Constructor
    /// @throw std::invalid_argument if type T doesn't implement the void clear() method
    OptimVector();

    /// @brief Move Constructor
    /// @param other Object to move
    OptimVector(OptimVector &&other);

    /// @brief Move assignment operator
    /// @param other Object to move
    OptimVector &operator=(OptimVector &&other);

    /// @brief Returns const iterator to the begin of the vector
    const_iterator_type cbegin() const;

    /// @brief Returns const next_it_ instead of the end of the vector,
    /// as if the items beyond this iterator have been cleared
    const_iterator_type cend() const;

    /// @brief Returns a reference to the element at specified location @p pos, with bounds checking
    /// @param pos Position of the element to return
    T &at(size_t pos);

    /// @brief Returns a const reference to the element at specified location @p pos, with bounds checking
    /// @param pos Position of the element to return
    const T &at(size_t pos) const;

    /// @brief Returns the number of elements in the container, i.e. std::distance(begin(), end())
    size_t size() const;

    /// @brief Returns whether the vector is empty (i.e. whether its size is 0).
    bool empty() const;

    /// @brief Makes a new element available at the end of the container
    /// @return Reference to this last element in the container
    T &allocate_back();

    /// @brief Makes all elements in the container unavailable by shifting the end ptr to begin
    void clear();

    /// @brief Increases the capacity of the vector to a value that's greater or equal to @p new_cap . If it's greater
    /// than the current capacity(), new storage is allocated, otherwise the method does nothing.
    void reserve(size_t new_cap);

    /// @brief Moves elements to another vector
    /// @param other Vector that will take ownership of the data
    void move_and_insert_to(OptimVector<T> &other);

    /// @brief Returns pointer to the underlying array serving as element storage
    T *data();

    /// @brief Returns const pointer to the underlying array serving as element storage
    const T *data() const;

private:
    /// @brief SFINAE test to make sure the clear method is implemented for the type T
    template<typename U>
    struct is_clear_impl {
        typedef char YesType[1];
        struct NoType {
            char dummy[2];
        };

        template<typename C>
        static YesType &test(decltype(&C::clear));

        template<typename C>
        static NoType &test(...);

        static const bool value = (sizeof(YesType) == sizeof(test<U>(0)));
    };

    typename std::vector<T>::iterator next_it_; ///< Actual end ptr of the class
    std::vector<T> vector_;                     ///< Base container, the size of which can only increase
};

template<typename T>
inline OptimVector<T>::OptimVector(OptimVector<T> &&other) {
    const auto dist = std::distance(other.vector_.begin(), other.next_it_);
    this->vector_   = std::move(other.vector_);
    this->next_it_  = this->vector_.begin();
    std::advance(this->next_it_, dist);
}

template<typename T>
inline OptimVector<T> &OptimVector<T>::operator=(OptimVector<T> &&other) {
    const auto dist = std::distance(other.vector_.begin(), other.next_it_);
    this->vector_   = std::move(other.vector_);
    this->next_it_  = this->vector_.begin();
    std::advance(this->next_it_, dist);
    return *this;
}

template<typename T>
inline OptimVector<T>::OptimVector() {
    if (!is_clear_impl<T>::value)
        throw std::invalid_argument("Unable to instantiate class OptimVector with a template "
                                    "type that doesn't implement a void clear() method.");
    next_it_ = vector_.end();
}

template<typename T>
inline typename OptimVector<T>::const_iterator_type OptimVector<T>::cbegin() const {
    return vector_.cbegin();
}

template<typename T>
inline typename OptimVector<T>::const_iterator_type OptimVector<T>::cend() const {
    return next_it_;
}

template<typename T>
inline T &OptimVector<T>::at(size_t idx) {
    if (idx + 1 > this->size())
        throw std::out_of_range("Index out of range.");
    return vector_[idx];
}

template<typename T>
inline const T &OptimVector<T>::at(size_t idx) const {
    if (idx + 1 > this->size())
        throw std::out_of_range("Index out of range.");
    return vector_[idx];
}

template<typename T>
inline size_t OptimVector<T>::size() const {
    return std::distance(cbegin(), cend());
}

template<typename T>
inline bool OptimVector<T>::empty() const {
    return cbegin() == cend();
}

template<typename T>
inline T &OptimVector<T>::allocate_back() {
    if (next_it_ == vector_.end()) {
        // increase the size by 1
        vector_.emplace_back();
        next_it_ = vector_.end();
        return vector_.back();
    } else {
        // first clear
        // and then pass reference to rewrite on it
        assert(0 <= this->size() && this->size() <= vector_.size());
        next_it_->clear(); // Warning this clear method must be defined for T
        T &ref = *next_it_;
        ++next_it_;
        return ref;
    }
}

template<typename T>
inline void OptimVector<T>::clear() {
    next_it_ = vector_.begin();
}

template<typename T>
void OptimVector<T>::reserve(size_t new_cap) {
    const auto dist = std::distance(vector_.begin(), next_it_);
    vector_.reserve(new_cap); // Reserve might invalidate next_it_
    next_it_ = vector_.begin();
    std::advance(this->next_it_, dist);
}

template<typename T>
void OptimVector<T>::move_and_insert_to(OptimVector<T> &other) {
    if (this->empty())
        return;

    other.reserve(other.size() + this->size());
    for (auto it = vector_.begin(); it != next_it_; it++) {
        auto &back = other.allocate_back();
        std::swap(back, *it);
    }
    this->clear();
    assert(0 <= this->size() && this->size() <= vector_.size());
}

template<typename T>
inline T *OptimVector<T>::data() {
    return vector_.data();
}

template<typename T>
inline const T *OptimVector<T>::data() const {
    return vector_.data();
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_OPTIM_VECTOR_H