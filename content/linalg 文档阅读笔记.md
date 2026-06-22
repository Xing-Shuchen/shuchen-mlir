---
tags: [ai-compiler]
created: 2026-04-29
---

按：这一篇是阅读 [`linalg` documents](https://mlir.llvm.org/docs/Dialects/Linalg/) 的笔记，逻辑结构遵循源文档内容，不是非常适合作为学习用。个人认为 `linalg` 方言是 MLIR 构建 AI 编译器时最重要的方言（可以没有之一）——模型在上层框架（TensorFlow、Torch）中都表示为计算图，计算图是一个有向无环图 DAG，其节点是操作而边是张量数据—— `linalg` 的完美嵌套循环假设和迭代空间隐式假设十分适合与连接 tensor 类型数据和底层标量数据，适合于描述张量运算这种具有非常规整的计算模式的运算；而且切分/tiling，融合/fusion 等重要变换的基础都是在 `linalg` 层。

# 1 Technical Stack

| Level abbr.                                  | Context/Definition                                 |
| -------------------------------------------- | -------------------------------------------------- |
| Operator Graph, OpGraph                      | 最高层：计算图 / 数据流图。                                    |
| Tensor-Structured Ops With Broadcasts, TSOWB | 仍保持高层张量结构，但广播、布局约束等更明确。                            |
| CodeGen-Aware Selection, CGASel              | 面向不同 codegen 路径进行选择的层。                             |
| High-level Hierarchical Optimization, HHO    | tiling / fusion 等高层层级优化所在层，**Linalg** 位于这里。        |
| Mid-level Hierarchical Abstraction, MHA      | 向 loops / memory layout 过渡的层，如 `Affine` / `SCF` 等。 |
| High-Level Target-Specific IR, HLTSIR        | 更贴近硬件向量与目标特性的层，如 **Vector**。                       |
| Target-Specific IR, TSIR                     | 接近最终机器码前的层，如 LLVM IR / SPIR-V。                     |

# 2 Rationale：逻辑依据与定位

`linalg` 设计出来主要是为了解决：

- 高级层级优化（High-level Hierarchical Optimization）；
- 与混合专家式编译环境（不同 codegen 路径）之间的协同问题。

一句话总结：

> **它的职责是在“还保留结构语义”时完成高价值优化。**

# 3 常见关键变换

（以下变换的设计理念与详细机制可以阅读 [Lei Zhang 大佬的这篇文章](https://www.lei.chat/zh/posts/mlir-linalg-dialect-and-patterns/)。）

1. Progressive Buffer Allocation：渐进式缓冲区分配；
2. Parametric Tiling：参数化分块；
3. Promotion to Temporary Buffer in Fast Memory：提升到快速内存中的临时缓冲区；
4. Producer-Consumer Fusion on Tiles：分块后的生产者-消费者融合；
5. Map to Parallel and Reduction Loops / Hardware：映射到并行与规约循环/硬件；
6. Vectorization：改写为 vector 形式；
7. Lower to Loops: lower 到 `Affine` / `SCF` / `cf` 等更低层控制流；
8. Lower to Library Calls / Intrinsics / ISA：lower 到库调用或硬件指令；
9. Partially Lower to Finer-Grained LinAlg Ops：部分 lower 成更细粒度的 `linalg` 操作。

# 4 `LinAlg` 操作的高阶描述

`LinAlg` 通过 `linalg.generic` 操作来使能自定义操作，`linalg.generic` 带有着关键变换，以及其他变换，比如 lower 到 scalar，甚至是外部 library 调用和 intrinsic 调用。

`linalg` 可以在张量或 buffer 上操作，输入 `ins` 输出 `outs`。

输出张量有两个功能：提供统一抽象、规定输出的尺寸。

`outs` 可以是**初始化张量**：`outs` 张量会提供输出张量的初始值，例如在执行矩阵乘法之前就已经有初值。

1. **破坏性更新**：张量不是一次性生成，而是在迭代中不断更新；编译器也会优化为在 buffer 中的**就地更新（inplace updating）**，而不是严格生成副本（优化手段之一，非严格要求）。
2. **具体存储（materialized）**：初始化张量总是在某种意义上被具体存储，如果优化的非常激进，则会变成完全保存在寄存器中的 SSA 值，根本不写回内存。

> [!note] materialize
> Specify abstract, implicit or logical values, loops and data structures to explicit values or structures operated in IR.
> 将抽象、隐式或逻辑上的值/循环/数据结构，具体化为 IR 中可操作的显式值或结构。

`outs`也可以是**shape-only** 张量：其内含的值没有被使用，这个张量只是为了携带尺寸信息，用于更低层级的优化 pass。计划：用一个专门的尺寸类型（shape type）来替代。

## 4.1 Payload-Carrying Ops 带负载操作
>
> [!quote]
> Linalg defines a payload carrying operation that implements the [structured op](https://docs.google.com/presentation/d/1P-j1GrH6Q5gLBjao0afQ-GfvcAeF-QU4GXXeSy0eJ9I/edit#slide=id.p) abstraction on tensors and buffers. This `linalg.generic` operation can express custom operations that optionally have _indexing semantics_ (by accessing the iteration indices using the `linalg.index` operation). The properties of `linalg.generic` are the result of applying the guiding principles described in the [Rationale Document](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/). They are listed next, with a brief example and discussion for each.

**带负载操作**：`LinAlg` 操作本身除了输入/输出的 tensor/buffer 之外，还携带一段**负载**——即内部的计算逻辑，表示操作在每个迭代元素或循环上的具体计算。

**结构化操作**：`LinAlg` 提供统一接口，对 tensor/buffer 做操作，循环展开、寄存器分配、内存访问的细节和优化用户在此阶段不必关心，最好对各种 tensor/buffer 操作有统一的表示。_带负载操作就实现了结构化操作这一抽象。_

**索引语义**：在元素计算内部可以 get 到当前迭代位置。`LinAlg` 操作能够访问迭代器索引（通过 `linalg.index` 操作），而非仅是张量元素；这使得操作能够实现自定义算子，能够实现带偏置卷积、坐标操作、位置编码等能力。

**Rationale Document Principles** 逻辑指导原则：可组合性 composability、结构化循环表示 structured loops、负载与 IR 分离 payload-IR independence、通用性 generality、显式索引 explicit indexing。

### 4.1.1 Property 1: 输入和输出操作数定义迭代空间

一个 `linalg.generic` 操作的迭代空间完全的由其操作数导出。局部 IR 单元（IR element）具有完整的循环控制流信息（迭代器数量、范围、并/串行、是否归约等等）。**每个操作是自包含的，可独立优化的 IR 单元。**

```mlir title='example1.mlir'
#accesses = [
  affine_map<(m) -> (m)>,
  affine_map<(m) -> (m)>
]

#attrs = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

func.func @example(%A: memref<?xf32, strided<[1]>>,
              %B: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
  linalg.generic #attrs
  ins(%A: memref<?xf32, strided<[1]>>)
  outs(%B: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
  ^bb0(%a: f32, %b: vector<4xf32>):
    %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
    linalg.yield %c: vector<4xf32>
  }
  return
}
```

Property 1 具体化为如下 `scf` 循环 IR：

```mlir title='scf of example1'
// Run: mlir-opt example1.mlir -allow-unregistered-dialect -convert-linalg-to-loops
// This converted representation is in the `scf` dialect.
// It's syntax can be found here: https://mlir.llvm.org/docs/Dialects/SCFDialect/

func.func @example(%arg0: memref<?xf32>,
                   %arg1: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf32>
  scf.for %arg2 = %c0 to %0 step %c1 {
    %1 = memref.load %arg0[%arg2] : memref<?xf32>
    %2 = memref.load %arg1[%arg2]
       : memref<?xvector<4xf32>, strided<[2], offset: 1>>
    %3 = "some_compute"(%1, %2) : (f32, vector<4xf32>) -> vector<4xf32>
    memref.store %3, %arg1[%arg2]
       : memref<?xvector<4xf32>, strided<[2], offset: 1>>
  }
  return
}
```

在没有 lower 到循环形式之前，循环归纳变量和迭代器都是隐式的（i.e., 尚未实质化），主要隐含的是：

1. 操作的语义限制于在结构化数据类型上操作，如 tensor/buffer，有 shape，仅在其上才能定义迭代器。
2. payload 逻辑只能建模纯计算操作，不能建模任意代码/副作用：全局变量写入、IO、打印、修改外部状态、破坏性修改非张量内存。保证了可优化性、语义确定性，保持了结构化操作的抽象。

> [!note] loop induction varaible 循环归纳变量
> 循环里那个“每轮按照固定规律变化”的变量。
>
### 4.1.2 Property 2: 控制结构和数据结构的反向映射

一个 `linalg.generic` 操作会定义**迭代空间和数据的互映射**。

```mlir title='example2.mlir'
#indexing_maps = [
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> (j)>
]

#attrs = {
  indexing_maps = #indexing_maps,
  iterator_types = ["parallel", "parallel"]
}

func.func @example(%A: memref<8x?xf32, strided<[2, 2], offset: 0>>,
              %B: memref<?xvector<4xf32>>) {
  linalg.generic #attrs
  ins(%A: memref<8x?xf32, strided<[2, 2], offset: 0>>)
  outs(%B: memref<?xvector<4xf32>>) {
  ^bb0(%a: f32, %b: vector<4xf32>):
    %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
    linalg.yield %c: vector<4xf32>
  }
  return
}
```

Property 2具体化为如下形式

```mlir title='scf of example2'
// Run: mlir-opt example2.mlir -allow-unregistered-dialect -convert-linalg-to-loops

func.func @example(%arg0: memref<8x?xf32, strided<[2, 2]>>, %arg1: memref<?xvector<4xf32>>) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c1 : memref<8x?xf32, strided<[2, 2]>>
  scf.for %arg2 = %c0 to %0 step %c1 {
    scf.for %arg3 = %c0 to %c8 step %c1 {
      %1 = memref.load %arg0[%arg3, %arg2] : memref<8x?xf32, strided<[2, 2]>>
      %2 = memref.load %arg1[%arg3] : memref<?xvector<4xf32>>
      %3 = "some_compute"(%1, %2) : (f32, vector<4xf32>) -> vector<4xf32>
      memref.store %3, %arg1[%arg3] : memref<?xvector<4xf32>>
    }
  }
  return
}
```

这个循环映射需要反转，因为我们想要实现如下的两种分析：

1. 给定迭代空间子集，要访问数据的什么子集？
2. **给定数据子集，如何确定迭代空间子集？**
实现第二个分析是 `LinAlg` 实现关键变换的基础，如[[#3 常见关键变换]]的2、3、4.

### 4.1.3 Property 3: 迭代器类型显式定义

一个 `linalg.generic` 操作完整地声明了迭代器类型。这个信息会用在关键变换中。

> [!example] Types of iterations
> parallel, reduction, partition, permutable/monotonic, sequential, dependence distance...
> At this time `LinAlg` only has an explicit use for _parallel_ and _reduction_.

在 `LinAlg` 层面声明迭代器类型可以传递在低层 IR 很难甚至无法分析出的属性。在 polyhedral 社区这个属性叫做”bands”。

另外，迭代器类型也是来自前端/用户的保证 guarantees，编译器可以利用此信息进行优化。

> [!example] histogram computation
> 最常见的例子是直方图计算。见[[MLIR LinAlg 直方图计算]]
>
### 4.1.4 Property 4: 计算负载在 Region 中指定

一个 `linalg.generic` 操作拥有完全通用的计算负载，得意于使用了 [Regions](https://github.com/llvm/llvm-project/blob/58265ad42a90ae8905be6a447cb42e53529a54a0/mlir/docs/LangRef.md/#regions)。Region 以 tensor/buffer 的 scalar 元素为参数，额外的值也可以通过参数传递，以使得 Region 可以调用库函数。

目前，对 Region 语义无额外约束。前端负责迭代器类型的语义以匹配 Region 内的操作：Region 可以任意读取-写入 buffer。_如果与某些 parallel 迭代器约束相冲突，这是 undefined behavior。_

下列代码用以展示此 Property：

```mlir title='example3.mlir'
#map = affine_map<(i, j) -> (i, j)>

#attrs = {
  indexing_maps = [#map, #map, #map],
  iterator_types = ["parallel", "parallel"]
}

func.func @example(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  linalg.generic #attrs
  ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
  outs(%C: memref<?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = arith.addf %a, %b : f32
      linalg.yield %d : f32
  }

  return
}
```

Property 4具体化为：

```mlir title='scf of example3'
func.func @example(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  scf.for %arg3 = %c0 to %0 step %c1 {
    scf.for %arg4 = %c0 to %1 step %c1 {
      %2 = memref.load %arg0[%arg3, %arg4] : memref<?x?xf32>
      %3 = memref.load %arg1[%arg3, %arg4] : memref<?x?xf32>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg2[%arg3, %arg4] : memref<?x?xf32>
    }
  }
  return
}
```

### 4.1.5 Property 5: 也许会映射到外部库调用

一个 `linalg.generic` 操作可能会映射到外部库调用，通过指定一个 `SymbolAttr`。在 `LinAlg` 这个抽象层级，我们可以做各种变换（平铺、融合等等），但要保证不破坏 generic 语义结构，保留结构信息，让外部库调用可行。

```mlir title='example4.mlir'
#indexing_maps = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i, j)>
]

#attrs = {
  indexing_maps = #indexing_maps,
  iterator_types = ["parallel", "parallel"],
  library_call = "pointwise_add"
}

func.func @example(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  linalg.generic #attrs
  ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
  outs(%C: memref<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %d = arith.addf %a, %b : f32
    linalg.yield %d : f32
  }
  return
}
```

Property 5 具体化为：

```mlir title='scf of example4'
func.func @example(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %0 = memref.cast %arg0 : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  %1 = memref.cast %arg1 : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  %2 = memref.cast %arg2 : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  call @pointwise_add(%0, %1, %2) : (memref<?x?xf32, strided<[?, ?], offset: ?>>,
    memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
  return
}
func.func @pointwise_add(memref<?x?xf32, strided<[?, ?], offset: ?>>,
                         memref<?x?xf32, strided<[?, ?], offset: ?>>,
                         memref<?x?xf32, strided<[?, ?], offset: ?>>) attributes {llvm.emit_c_interface}
```

最终，转化为以下 LLVM IR：

```mlir title='llvm ir of example4'
func.func @example(%arg0: !llvm<"float*">, ...) {
  ...
  llvm.call @pointwise_add(...) : (!llvm<"float*">, ...) -> ()
  return
}

llvm.func @pointwise_add(%arg0: !llvm<"float*">, ...) attributes {llvm.emit_c_interface} {
  ...
  llvm.call @_mlir_ciface_pointwise_add(%9, %19, %29) : (!llvm."{ float*, float*, i64, [2 x i64], [2 x i64] }*">, !llvm<"{ f32*, f32*, i64, [2 x i64], [2 x i64] }*">, !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }
*">) -> ()
  llvm.return
}
llvm.func @_mlir_ciface_pointwise_add(!llvm."{ float*, float*, i64, [2 x i64], [2 x i64] }*">, !llvm<"{ f32*, f32*, i64, [2 x i64], [2 x i64] }*">, !llvm<"{ f32*, f32*, i64, [2 x i64], [2 x i64] }*">) attributes {llvm.emit_c_interface}
```

#### 4.1.5.1 外部库调用约定

`LinAlg` 方言采用了类似 `BLAS` 的调用约定：把计算卸载到快速库函数实现时，传递带有元数据的非拥有（non-owning）指针。`MKL, OpenBLAS, BLIS, cuBLAS, cuDNN` 都采取了类似的调用约定。一般地，`LinAlg` 传递 View 数据结构的非拥有指针给预编译外部链接库。

### 4.1.6 Property 6: 完美嵌套地写入整个输出操作数

完美嵌套循环是结构化数据的重要属性，使能了平铺和外部调用。不幸的是，这玩意很容易被破坏，平铺和外部调用就变得很难甚至不可能。`LinAlg` 将完美嵌套属性作为一等公民，要求结构不能被破坏。这是 `LinAlg` 在设计之初就纳入保证的。

需要指出的是，非完美嵌套循环可以转换为完美嵌套循环，通过充分的循环体拆分和最内层循环的条件语句。例子见[[非完美嵌套循环变换]]。基本技巧是拆分循环体和将非仿射控制流转换为仿射计算依赖，把控制依赖改写成数据依赖/谓词表达。

另外，深度谓词变换还需在 `LinAlg` 的变换结束后变换回去，在迭代器和归纳变量具体化后，上述变换会极大影响正规化（canonicalization）的质量、折叠和循环不变代码移动（Loop Independent Code Motion，LICM），从而降低整体的性能。

### 4.1.7 总论

上述六个属性定义了 `linalg.generic` 操作的语义。在实践中，这些语义是严格需要的，还是一些应当或可以自动导出的，仍然是一个开放问题。

至少目前，多个高级的编译器实践经验表明，采用以上属性的组合是个不错的方案。

## 4.2 Data Representation: Views

当前的 `LinAlg` 实现使用了带步长的内存引用（[Strided MemRef (a.k.a View)](https://groups.google.com/a/tensorflow.org/forum/#!topic/mlir/MaL8m2nXuio)）抽象。未来可能使用其他数据结构，支持参差张量（ragged），混合稀疏张量（mixed-sparse）类型的数据。

> [!explanation] ragged, mixed-sparse tensors
> ragged tensor 可以是：
> $$
> \begin{align}
> &[[1,2,3] \\
> &[4] \\
> &[5,6]]
> \end{align}
> $$
> 这里每一行长度不同，所以不能简单看作一个标准的 `3 x ?` 矩形数组。
>
> mixed-sparse 张量：部分维度稀疏，部分维度稠密，或者是一部分用 CSR，一部分用 COO，某些块是 dense block，某些块是 sparse block。
>
## 4.3 Metadata Ops 元数据操作

一些操作可以操纵元数据但不移动内存。这些操作操作数是 `View` + 额外属性，返回新 `View`。

目前是这些操作：

```mlir
memref.view
memref.subview
memref.transpose
linalg.slice
linalg.reshape
```

以后打算加入：

```mlir
linalg.tile
linalg.intersection
linalg.convex_union
linalg.difference
```

这些操作是为了支持大规模分布式邻域网格计算。

远景规划中，Legion 数据中心编程模型也将得到支持。

## 4.4 Named Payload-Carrying Ops 具名带负载操作

`LinAlg` 提供了通过具名操作的一个小子集：`linalg.fill, linalg.dot, linalg.matmul, linalg.conv`。这些具名操作导向 `linalg.generic` 操作。其他常用具名操作如 `linalg.reduce`、`linalg.broadcast` 见 [[linalg Dialect]]。

正在开发：声明式机制来自动生成具名操作。通过描述通用操作结构的组合来通过 `Tablegen` 自动生成具名操作。

## 4.5 Named Payload Ops Specification 具名操作规范

`LinAlg` 提供一个声明式规范和一个生成工具 `mlir-linalg-ods-gen`，从一种类似于爱因斯坦求和约定的记号系统中自动生成具名操作。

`mlir-linalg-ods-gen` 的语法借鉴了 Tensor Comprehensions（TC），有些许改动：

1. 输入输出张量参数的规范是 `id : type(symbolic-affine-expression-list)`，每个新符号都采用急切发现机制（discovered eagerly）。TC 不支持通用符号化仿射表达式。
2. 输出尺寸显示指定。TC 由输入导出。
3. 只允许写几种固定的运算构造子；生成器一看见这些构造子，就直接产出对应的 builder 调用。
4. 归约维度用尖括号指定。TC 中归约维度是推断得到的。
5. 平行和归约维度以文本程序排序。
6. 操作可以使用 `attract( strides:2xi32 )` 来定义一系列属性。
这些语法都在演进中。

当前语法和语义的限制有：

1. 每一个 `def` 只能包含一个综合 comprehension，但每个综合可以执行多个更新。
2. 每个张量可能只能用单一索引表达式来使用。

用 `"""` 包起来的字符文本可以添加到一个操作里，首行是一行总结性描述，后面可以跟更长的详细描述。

## 4.6 YAML Based Named Structure Ops 基于 YAML 的具名结构化操作

`LinAlg` 提供声明式生成工具 `mlir-linalg-ods-yaml-gen`，从操作的 YAML 格式描述中自动生成具名操作。YAML 操作描述从一个高级 DSL 生成，并不能直接修改。
