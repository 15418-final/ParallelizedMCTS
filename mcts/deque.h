#ifndef DEQUE_H
#define DEQUE_H

#define BDSIZE 19

#include <cuda.h>
#include <cuda_runtime.h>

template <class T>
class Deque {
private:
	T* data;
	int head;
	int _size;
	int _capacity;

public:
	class iterator {
	private:
		int _ptr;
		Deque& container;
	public:
		__device__  __host__ iterator(int h, Deque& outer): _ptr(h), container(outer){}

		__device__  __host__ iterator operator++() {
			iterator old = *this;
			_ptr = (_ptr + 1) % container._capacity;
			return old;
		}
		__device__  __host__ iterator operator++(int r) {
			_ptr = (_ptr + 1) % container._capacity;
			return *this;
		}
		__device__  __host__ T& operator*() {
			return container.data[_ptr];
		}
		__device__  __host__ bool operator==(const iterator& rhs) { return _ptr == rhs._ptr; }
		__device__  __host__ bool operator!=(const iterator& rhs) { return _ptr != rhs._ptr; }
	};

	__device__ __host__ Deque() {
		_capacity = BDSIZE * BDSIZE + 1;
		data = static_cast<T*>(malloc(_capacity * sizeof(T)));
		head = 0;
		_size = 0;
	}

	__device__ __host__  ~Deque() {
		free(data);
	}

	__device__ __host__ T& operator[] (const int index) {
		int pos = (head + index) % _capacity;
		return data[pos];
	}

	__device__ __host__ void push_back(const T& e) {
		// Make sure head and tail won't point to same position. Good for implementing iterator
		if (_size < BDSIZE * BDSIZE - 1) {
			int tail = (head + _size) % _capacity;
			data[tail] = e;
			_size++;
		}
	}

	__device__ __host__  T front() {
		if (_size > 0) {
			return data[head];
		}
		return T();
	}

	__device__ __host__  T pop_front() {
		if (_size > 0) {
			T e = data[head];
			head = (head + 1) % _capacity;
			_size--;
			return e;
		}
		return T();
	}

	__device__ __host__  T back() {
		if (_size > 0) {
			int tail = (head + _size) % _capacity;
			return data[tail];
		}
		return T();
	}

	__device__ __host__  T pop_back() {
		if (_size > 0) {
			int tail = (head + _size) % _capacity;
			T e = data[tail];
			_size--;
			return e;
		}
		return T();
	}

	__device__ __host__  int size() {
		return _size;
	}

	__device__ __host__  iterator begin() {
		return iterator(head, *this);
	}

	__device__ __host__  iterator end() {
		int tail = (head + _size) % _capacity;
		return iterator(tail, *this);
	}

	__device__  __host__ void clear() {
		head = 0;
		_size = 0;
	}

};


#endif