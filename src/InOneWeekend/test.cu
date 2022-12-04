#include<iostream>
using namespace std;

class base {
public:
	__host__ __device__ virtual void print() = 0;

	__host__ __device__ void show()
	{
		printf("show base class\n");
	}
};

class derived : public base {
public:
	__host__ __device__ void print()
	{
		printf("print derived class\n");
	}

	__host__ __device__ void show()
	{
		printf("show derived class\n");
	}
};

class testclass {
    public:
    __device__ void whoami() {
		printf("test class\n");

    }
};

__device__ void test(base **array) {
    base *d = array[0];
    d->print();
	d->show();
}

__global__ void init_array(testclass **obj) {
    *obj = new testclass;
}

__global__ void kernel(testclass **obj) {
    // derived **array;
    // cudaMalloc(&array, sizeof(derived*) * 500);

	// derived *d = new derived;

    // //array[0] = d;

	// d->print();
	// d->show();
    // array[0]->print();
	// array[0]->show();

    //test(array);

    testclass **array = (testclass**) malloc(sizeof(testclass*) * 500);
    //testclass *t = new testclass;
    testclass *t = *obj;
    printf("Sizeof t: %d\n", sizeof(t));
    t->whoami();

    array[0] = t;
    array[0]->whoami();
    printf("Done!!\n");
}

int main()
{

    base **array;
    cudaMallocManaged(&array, sizeof(base*) * 300);

	// // Virtual function, binded at runtime
	// bptr->print();

	// // Non-virtual function, binded at compile time
	// bptr->show();
    testclass **world;
    cudaMalloc(&world, 500*sizeof(testclass *));
    init_array<<<1,1>>>(world);

    cudaDeviceSynchronize();
    kernel<<<1,1>>>(world);
	
    cudaDeviceSynchronize();
	return 0;
}
