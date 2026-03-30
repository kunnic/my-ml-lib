1. C memory model:
    - Stack, heap allocation and when to use each.
        + Stack:
            - Stack is a place where computer stores address of the code's data, using LIFO logic to store the code logic.
        + Heap:
            - Heap is the place where it stores actual data, but needs to be declared and allocated
    - malloc, calloc, free:
        + malloc: checks for avalability and returns a pointer points at the first data block of giver data type. Uninit data block returns NULL or garbage value.
        + calloc: checks for avalability and returns a pointer points at the first data block of giver data type. Its data blocks are all init with 0's.
        + free: free the given data which were allocated.
    - Struct:
        + Work kinda like modern class
        + -> works the same for get() in OOP. only get struct's data.
    - Memory leak, double-free and use-after-free
        + Memory leak: when the allocated data block is done using but was not freed, pile up after time and make the memory full of unfreed data, it doesn't allow new app access the current data because it's not freed, causing memory overflow.
        + Double-free: call free 2 times make 2 free state of the same data block in the compter, so it can be allocated 2 times causing fatal error on both codes. causing crashes because the data is corrupted, for example: the program A use data types B on the allocated block, but program C use data type D instead of A, causing fatal error.
        + -fsanitize=address works like an Exception when  compoling these kind of code.