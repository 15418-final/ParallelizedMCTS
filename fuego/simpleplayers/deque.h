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
		int tail;
		int _size;


	public:
		__device__ __host__ Deque() {
			data = static_cast<T*>(malloc(BDSIZE*BDSIZE*sizeof(T)));
			head = 0;
			tail = 0;
			_size = 0;
		}

		__device__ __host__  ~Deque() {
			free(data);
		}

		__device__ __host__ T& operator[] (const int index) {
			return data[index];
		}

		__device__ __host__ void push_back(const T& e) {
			if (tail < BDSIZE*BDSIZE - 1) {
				data[tail++] = e;
				_size++;
			}
		}

		__device__ __host__  T pop_front() {
			if (head != tail) {
				T e = data[head++];
				_size--;
				return e;
			}
			return NULL;
		}

		__device__ __host__  T pop_back() {
			if (head != tail) {
				T e = data[tail--];
				_size--;
				return e;
			}
			return NULL;
		}

		__device__ __host__  int size() {
			return _size;
		}

		__device__ __host__  int begin() {
			return head;
		}

		__device__ __host__  int end() {
			return tail;
		}

		__device__  __host__ void clear() {
			head = 0;
			tail = 0;
			_size = 0;
		}
};


#endif