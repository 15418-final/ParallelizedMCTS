#ifndef DEQUE_H
#define DEQUE_H

#define BDSIZE 19


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
		iterator(int h, Deque& outer): _ptr(h), container(outer) {}

		iterator operator++() {
			iterator old = *this;
			_ptr = (_ptr + 1) % container._capacity;
			return old;
		}
		iterator operator++(int r) {
			_ptr = (_ptr + 1) % container._capacity;
			return *this;
		}
		T& operator*() {
			return container.data[_ptr];
		}
		bool operator==(const iterator& rhs) { return _ptr == rhs._ptr; }
		bool operator!=(const iterator& rhs) { return _ptr != rhs._ptr; }
	};

	Deque() {
		_capacity = BDSIZE * BDSIZE + 1;
		data = static_cast<T*>(malloc(_capacity * sizeof(T)));
		head = 0;
		_size = 0;
	}

	~Deque() {
		free(data);
	}

	T& operator[] (const int index) {
		return data[index];
	}

	void push_back(const T& e) {
		// Make sure head and tail won't point to same position. Good for implementing iterator
		if (_size < BDSIZE * BDSIZE - 1) {
			int tail = (head + _size) % _capacity;
			// printf("tail:%d\n",tail);
			data[tail] = e;
			_size++;
			// printf("size:%d\n",_size);
		}
	}

	T front() {
		if (_size > 0) {
			return data[head];
		}
		return NULL;
	}

	T pop_front() {
		if (_size > 0) {
			T e = data[head];
			head = (head + 1) % _capacity;
			_size--;
			return e;
		}
		return NULL;
	}

	T back() {
		if (_size > 0) {
			int tail = (head + _size) % _capacity;
			return data[tail];
		}
		return NULL;
	}

	T pop_back() {
		if (_size > 0) {
			int tail = (head + _size) % _capacity;
			T e = data[tail];
			_size--;
			return e;
		}
		return NULL;
	}

	int size() {
		return _size;
	}

	iterator begin() {
		return iterator(head, *this);
	}

	iterator end() {
		int tail = (head + _size) % _capacity;
		return iterator(tail, *this);
	}

	void clear() {
		head = 0;
		_size = 0;
	}

};


#endif