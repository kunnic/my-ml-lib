Step 1.0:

- Can explain stack vs heap: when each is used, pros/cons:
    + Stack:
        - When a function is called, system packs its needed info in a block called "stack frame" and add it on the top of the stack. When the function have done its job, it'd be poped out and its allocated memory is freed.
        - Stack is controlled by:
            + Instruction Pointer: points to the next instruction for CPU to run
            + Base pointer: points to the bottom of the current stack frame
            + Stack frame: function's content
                - Return address of the caller function for Instruction Pointer
                - Parameters of the current function
                - Local variables of the current function
                - Logic base pointer that points to its previous stack frame's base pointer
            + Stack pointer: points to the top of the stack
        - Pros:
            + Fast (Cache-friendly)
            + Automatically clean because of the nature of stack data structure
        - Cons:
            + Small size limit
            + Unresolved overflow attack on x86 CPU
    + Heap:
        - When a dynamical memory allocation is called, the system allocates memory to the heap memory location. When the memory block have done its job, the memory must be freed for further purpose.
        - Pros:
            + As big as RAM can provide
        - Cons:
            + Easily lead to memory-related fatal error
            + Fragmentation


