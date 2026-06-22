MLIR 框架代码中包含了48个方言，这些方言被称为 core and contributed，是 MLIR 多级 IR 体系的内容。

MLIR 是编译器基础设施，尤其用于构建一类非常注重中端优化的编译器即机器学习编译器/ AI 编译器。与传统编译器一样，可以划分为前中后三个阶段。前端负责承接“高级语言”，中端负责通用计算和优化，后端负责生成“低级语言”，按照这个逻辑，内置的方言可以分为三类：前端对接类方言承接上层模型计算图或编程框架，中端优化类负责进行各种分析、变换与优化，后端生成类对接更底层 IR 以至于不同架构的汇编语言。

另外，还有些方言不参与上述三类任一功能，而是穿透三类或者是描述与计算无关的信息或元数据，这些方言单独划归一类，称为计算无关方言。

以下按照三类进行鸟瞰，每类内部以字典序排布。

# 计算无关方言

## dlti Dialect
`dlti` 即 Data Layout and Target Information，该方言不参与计算，职能是在 IR 中以结构化方式携带目标平台相关参数，供其他方言在 lowering 过程中查询。`dlti` 提供的是一套*属性/attribute* 机制：`#dlti.dl_spec` 聚合多个 `#dlti.dl_entry` 条目，以键值对形式描述 layout 和 target 信息，如位宽、ABI 对齐、指针大小、端序等等。这些 specification 通过 `DataLayoutSpecInterface` 等接口附着到 [[#Builtin Dialect]] 的 `ModuleOp` 上并支持*作用域嵌套/scoped*，内层可以覆盖外层。除 layout 外，`dlti` 还能携带更一般的 target 信息，如设备标识、特性键值，服务异构编译场景。

`dlti` 与 [[#llvm Dialect]] 的 data layout 字符串相对应，是 [[#memref Dialect]] 计算字节布局、各后端方言做对齐、ABI 决策时的同一信息来源。

## index Dialect
`index` 是对*机器无关索引类型*上算术的抽象，专门服务 [[#Builtin Dialect]] 的 `index` 类型——一种位宽由目标平台决定（运行时等同 `size_t` / 指针宽度）的整数，广泛用作循环归纳变量、[[#memref Dialect]] 与 [[#tensor Dialect]] 的下标和维度尺寸。它补齐了 [[#arith Dialect]] 的空白：[[#arith Dialect]] 面向*固定位宽*的 `i32` / `i64` 等，而 `index` 方言提供专门作用于 `index` 类型的 `index.add` / `index.sub` / `index.mul` / `index.divs` / `index.divu`、比较 `index.cmp`、位运算，以及与定宽整数互转的 `index.casts` / `index.castu`。

`index` 特别之处是*位宽多态*：同一段 `index` 运算在 lowering 时按目标平台落为相应宽度，如 `index.constant` 可同时给出 32 位与 64 位两种取值供后端择，从而让涉及尺寸/下标的计算保持目标无关，直到最终降到 [[#llvm Dialect]] 才绑定具体位宽。

## irdl Dialect
`irdl` 方言即 *IR Definition Language*，是一个用以*在 IR 内部描述方言本身*的元层方言——它不表达运行时计算，而把「方言、操作、类型、属性的结构与约束」表示为 MLIR 操作，使方言定义成为可被加载、检查乃至*运行时动态注册*的一等数据，与传统借助 TableGen 在*编译期*静态生成 C++ 的路径形成对照。其结构 Op 层层嵌套：`irdl.dialect` 容纳若干 `irdl.operation` / `irdl.type` / `irdl.attribute`，而 `irdl.operands` / `irdl.results` / `irdl.parameters` 声明签名，约束则由 `irdl.is`（固定类型）、`irdl.any_of`（可选集合）、`irdl.any`、`irdl.parametric` 等*约束 Op* 组合表达，刻画操作数与结果须满足的类型条件。

`irdl` 的价值在于让方言可以*以数据形式定义*，支撑动态方言注册、方言互操作的形式化描述，以及对方言生态自身的工具化分析，是 MLIR「用 MLIR 描述 MLIR」自反射能力的集中体现。

## pdl_interp Dialect
`pdl_interp` 方言即 *PDL Interpreter*，是 [[#pdl Dialect]] 的*低层、可执行对应物*——如果说 `pdl` 以声明式描述「要匹配什么样的 IR 模式、重写成什么」，那么 `pdl_interp` 就是这套声明被*降级(lower)* 后得到的*命令式匹配字节码*，交由 PDL 字节码*解释器(interpreter)* 执行实际的图匹配与改写。它把高层的模式声明拆解为一连串细粒度的判定与遍历指令:`pdl_interp.get_operand` / `pdl_interp.get_result` / `pdl_interp.get_attribute` 等抽取被检查操作的组成，`pdl_interp.check_operation_name` / `pdl_interp.check_operand_count` / `pdl_interp.check_attribute` / `pdl_interp.are_equal` 等做*谓词判定*并据结果分支，`pdl_interp.record_match` 登记一次成功匹配，`pdl_interp.replace` / `pdl_interp.create_operation` 等执行重写。这种分解使匹配过程可被组织成高效的*决策树/状态机*——多条重写模式的公共前缀判定得以共享，避免逐条朴素重试。

开发者几乎从不手写 `pdl_interp`，它是编译 `pdl` 模式的中间产物，二者共同构成 MLIR *声明式重写基础设施(declarative rewrite infrastructure)* 的两个层次。

## ub Dialect
`ub` 方言即 *Undefined Behavior*,是对*未定义行为*的显式建模,**归入计算无关方言**——它不实现计算,而为「程序中本无定义、可由编译器任取」的情形提供一等表示。核心 Op 是 `ub.poison`:产出一个*毒值(poison value)*,表示一个未定义、编译器可自由假定的值,对应 LLVM IR 的 `poison`。把 UB 从「隐式约定」提升为「显式 IR 操作」的意义在于:许多变换会引入未定义结果(如越界、溢出占位、无效转换的占位结果),有了 `ub.poison` 即可*精确、可分析地*表达这种「这里的值无所谓」,而非靠隐含假设,既利于优化(编译器知道可自由处置),也利于正确性推理。它与 [[#arith Dialect]] 的 `nsw` / `nuw` 等溢出标志、[[#llvm Dialect]] 的 poison/undef 语义协同,是 MLIR 把 C/LLVM 式未定义行为规范化的设施。

## AttributeInterface definitions
**归入计算无关方言**(严格说这并非一个方言,而是一组横切设施的文档章节)。此处汇集 MLIR 中的 *Attribute Interface*——即定义在*属性(attribute)* 上的接口。属性是 MLIR 用以承载*编译期常量元数据*的机制(如类型、数值常量、布局规格、各类标记),而 *attribute interface* 让不同方言定义的属性能以*统一抽象*被通用代码查询与操作,无需硬编码具体属性类型。典型如 [[#dlti Dialect]] 借 `DataLayoutSpecInterface` 暴露数据布局、各类 `ElementsAttr` 接口统一访问张量常量数据、`MemRefLayoutAttrInterface` 抽象 [[#memref Dialect]] 的 layout。它体现 MLIR *面向接口而非面向具体类型* 的可扩展设计哲学,是支撑方言间互操作的元数据契约层。

## Builtin Dialect
`builtin` 方言**归入计算无关方言中的基础设施核心**——它是 MLIR 中唯一*总是被加载*的方言,提供整个体系赖以运转的最基础类型、属性与少量结构操作,本身不表达具体计算。它定义了几乎所有方言共用的*内建类型*:整数 `IntegerType`(`i32`、`i1` 等)、浮点 `FloatType`(`f16`/`f32`/`bf16` 等)、`IndexType`、聚合的 `VectorType` / `TensorType`(`RankedTensorType` 及 unranked)/ `MemRefType`、`ComplexType`、`FunctionType`、`TupleType` 等;以及*内建属性*:`IntegerAttr` / `FloatAttr` / `StringAttr` / `ArrayAttr` / `DictionaryAttr` / `DenseElementsAttr`(稠密张量常量)/ `TypeAttr` / `SymbolRefAttr` 等。结构上它提供顶层容器 `builtin.module`(`ModuleOp`,所有 IR 的根容器)与 `builtin.unrealized_conversion_cast`(渐进式 lowering 中跨方言类型的临时占位转换)。作为所有方言共享的*类型与属性词汇表*,`builtin` 是 MLIR 一切表示的公共底座。

## OpInterface definitions
**归入计算无关方言**(同样是一组横切设施的文档章节而非方言)。此处汇集 MLIR 的 *Operation Interface*——定义在*操作(operation)* 上的接口,是 MLIR 可扩展性的支柱机制:通用算法不针对具体 Op 编码,而面向接口编程,凡实现该接口的 Op(无论来自哪个方言)皆可被统一处理。典型接口如 `CallOpInterface` / `CallableOpInterface`(抽象调用与可调用体,见 [[#func Dialect]])、`LoopLikeOpInterface`(统一 [[#scf Dialect]] / [[#affine Dialect]] 的循环)、`RegionBranchOpInterface`(带区域的控制流)、`MemoryEffectsInterface`(声明读写/分配等*副作用*,驱动 DCE、调度等优化)、`InferTypeOpInterface`(类型推断)、`DestinationStyleOpInterface`(刻画 [[#linalg Dialect]] 的目的地传递风格)。正是这套接口让 MLIR 的 pass 与分析能跨方言复用,是其「一套基础设施容纳众多方言」得以成立的根本设计。

## pdl Dialect
`pdl` 方言即 *Pattern Descriptor Language*，是对*重写模式(rewrite pattern)本身*的声明式抽象——它让「匹配什么样的 IR、改写成什么」这件事不再用 C++ 手写 `RewritePattern`，而是表示为 MLIR 操作，成为可被加载、组合乃至*运行时动态注册*的一等数据，与 [[#irdl Dialect]] 用 IR 描述方言定义的自反射思路一脉相承(一个描述「方言长什么样」，一个描述「如何重写 IR」)。一个 `pdl.pattern` 内分两部分:匹配段用 `pdl.operation` / `pdl.operand` / `pdl.result` / `pdl.type` / `pdl.attribute` 声明待匹配的 IR 结构与约束，`pdl.rewrite` 段则描述命中后的改写动作，可借 `pdl.replace`、`pdl.operation`(构造新操作)等完成，并能调用外部 *native constraint / native rewrite* 处理纯声明难以表达的逻辑。它刻意只描述*模式*而不规定*执行策略*，真正的匹配由其降级所得的 [[#pdl_interp Dialect]] 字节码经解释器高效执行。

`pdl` 还是 [[#transform Dialect]] 等设施定位目标操作的底层匹配语言，二者(pdl / pdl_interp)共同构成 MLIR *声明式重写基础设施*的两个层次——`pdl` 面向人书写、`pdl_interp` 面向机器执行。

## smt Dialect
`smt` 方言是对 *SMT(Satisfiability Modulo Theories,可满足性模理论)* 公式的抽象,**归入计算无关方言**——它不表达运行时计算,而把逻辑约束建模为 IR,用以对接 Z3 等 SMT 求解器,服务形式化验证、等价性检查等。其类型有 `!smt.bool`、`!smt.bv<N>`(位向量)、`!smt.int` 等;操作覆盖各理论:布尔逻辑 `smt.and` / `smt.or` / `smt.not`、位向量运算、`smt.eq` / `smt.distinct`、量词 `smt.forall` / `smt.exists`,并以 `smt.assert` 断言公式、`smt.solver` 区域组织一次求解、`smt.check` 触发判定。它把约束求解能力引入 MLIR,使「证明某变换正确」「判定两段 IR 等价」等可在框架内表达,是偏验证/分析方向的元层方言。

## Tensor Operator Set Architecture (TOSA) Dialect
`tosa` 方言即 *Tensor Operator Set Architecture*,是一套*硬件无关、规格严谨的 ML 算子集*抽象,定位于前端对接类——它为各深度学习框架(TensorFlow、PyTorch、TFLite 等)提供一个*标准化、语义明确*的算子中间层,以收敛纷繁的框架算子定义、保证跨后端的可移植与数值一致性。其设计特点是算子集*精简且规格化*:每个算子(如 `tosa.conv2d`、`tosa.matmul`、`tosa.fully_connected`、`tosa.add`、`tosa.reduce_sum`、`tosa.clamp`、`tosa.reshape` 等)都有严格的形状、数据类型与数值行为定义,并原生考虑*量化*(与 [[#quant Dialect]] 协同)与整数/浮点 *profile*(面向不同硬件能力分级)。作为前端汇聚点,`tosa` 向下主要*合法化(legalize)* 到 [[#linalg Dialect]](%60tosa-to-linalg%60)进入通用优化与 lowering 管线,也可降到 [[#arith Dialect]] / [[#tensor Dialect]] / [[#scf Dialect]] 等,是 ML 模型进入 MLIR 中端的重要标准入口,与同为前端的 [[#linalg Dialect]] 形成「标准算子集 vs 结构化代数」的互补。

# 前端方言
## acc Dialect
`acc` 方言用以对接 OpenACC 编程框架。OpenACC 是一套基于*指令/ directive*的并行编程模型，用以方便的将现有 CPU 代码迁移到 GPU/加速器。

`acc` 方言保留了 OpenACC 的 gang/worker/vector 三层并行模型，忠实对应于 OpenACC 代码；每种 OpenACC 字句都对应独立的 `acc` 方言 Op。

## func Dialect
`func` 是对函数 function 这一基本程序结构的抽象，提供函数的定义、调用与返回能力，是 IR 中承载顶层可执行逻辑的标准容器。核心 Op 是 `func.func`，一个带符号名、签名、与函数体 Region 的可调用单元，通常嵌入于 [[#Builtin Dialect]] 的 `ModuleOp` 之内；`func.call` 通过符号引用做*直接调用*，`func.call_indirect` 借助函数值做*间接调用*；`func.return` 返回调用结果；`func.constant` 取得函数的引用值。函数签名以 builtin 的 `FunctionType` 表达，并通过 `CallOpInterface` / `FunctionOpInterface` 等接口与调用约定、内联、符号解析等通用机制对接。

`func` 是从 [[#scf Dialect]] / [[#cf Dialect]] 等控制流到模块组织之间的纽带，在 lowering 时其函数与调用语义自然映射到 [[#llvm Dialect]] 的 `llvm.func`、`llvm.call`，是贯穿前中后端、最基础的方言之一。

## ml_program Dialect
`ml_program` 方言是对*机器学习程序模块级结构*的抽象，为 ML 模型提供一套与具体计算无关的*程序组织与状态封装*设施，定位在前端计算图与 [[#linalg Dialect]]、[[#tensor Dialect]] 等实际计算方言之间，目标是给跨框架(TensorFlow、PyTorch 等)的模型 import 提供统一的模块级容器。它关注的不是「怎么算」而是「程序如何组织」：`ml_program.subgraph` / `ml_program.func` 表达子图与函数，`ml_program.global` 声明*模块级全局变量*——典型如神经网络的*权重/参数*这类需要在多次调用间持久存在的状态，配合 `ml_program.global_load` / `ml_program.global_store`(及其 `_graph` / `_const` 变体)读写，并以 `mutability`(可变/不可变)区分常量权重与可更新状态。这种对*全局状态*的显式建模，正是 ML 模型区别于普通函数式计算图的关键特征，使有状态推理与训练得以在 IR 层规范表达。

## omp Dialect
`omp` 对接 OpenMP 编程框架，与 [[#acc Dialect]] 同为*指令式/directive*的并行模型承接层，是 Flang 及 Clang 实现 OpenMP 的核心中介。既覆盖传统的*多核 CPU 共享内存并行*，也覆盖 *target offload（向 GPU、加速卡卸载）* 的并行。`omp` 忠实保留 OpenMP 的构造与子句：`omp.parallel` 开启并行区域、`omp.wsloop` / `omp.loop` 表达工作共享循环、`omp.sections` / `omp.single` / `omp.master` 等表达任务划分,`omp.task` 表达*任务(task)* 并行,`omp.target` / `omp.teams` / `omp.distribute` 表达卸载与设备端的 *teams / threads* 层级;同步上有 `omp.barrier` / `omp.critical` / `omp.atomic.*`,数据环境则以 `reduction`、`map`(host-device 数据映射)等子句和配套 Op 表达。

`omp` 与 [[#acc Dialect]] 形成对照：`acc` 用 *gang/worker/vector* 三层模型，它用 *teams/threads* 模型，但二者在 Flang 中可以并行发展，相互验证。在 lowering 路径上，`omp` 经 OpenMP 运行时（如 libomp、linomptarget）降到 [[#llvm Dialect]]，卸载路径则与 [[#gpu Dialect]] 等协同，是 MLIR 表达*主流共享内存*与*卸载并行*的关键前端方言。

## Transform Dialect
`transform` 方言是对*编译器变换调度(transformation schedule)本身*的抽象,**归入计算无关方言**——它不表示被编译的程序,而把「对 IR 施加哪些变换、以何顺序、作用于哪些目标」表示为另一段可被执行的 IR,即把传统硬编码在 C++ pass pipeline 里的优化策略提升为*可编程、可组合的一等数据*。其运作方式是:transform 操作作用于代表*待变换 payload IR* 中操作的*句柄(handle,类型如 `!transform.any_op`)*,通过 `transform.structured.match` 等定位目标(底层借助 [[#pdl Dialect]] 匹配),再以 `transform.structured.tile` / `transform.structured.fuse` / `transform.structured.vectorize` 等施加变换并产出新句柄,串联成完整 *schedule*。这使得 [[#linalg Dialect]] 的 tiling / fusion / vectorization 等优化得以*在 IR 中显式编排*而非埋于编译器代码,便于自动调优(autotuning)与策略搜索,是 MLIR 实现「schedule 与 computation 分离」(呼应 Halide/TVM 思想)的关键设施。

# 中端方言
## affine Dialect
`affine` 方言是对仿射操作和分析的抽象。`affine` 方言有坚实的数学基础——多面体编译（Polyhedral Compilation）理论，多面体要求循环边界和数组访问下表必须是循环变量与符号常量的仿射函数，使得编译器能够进行精确地依赖分析和变换。

`AffineMap` 是 MLIR 最重要的数据结构之一，描述从维度 + 符号 → 结果的仿射映射，描述了 MLIR 结构化（Structured）操作的迭代维度与迭代方式。

## arith Dialect
`arith` 方言对*基本标量与向量算术*抽象，提供整数/浮点数的四则、比较、位运算、类型转换等操作，是几乎所有方言都需要使用的计算原语。`arith` 方言设计为只承载 side-effect free/*无副作用*的纯运算：`arith.addi`/`arith.muli`、`arith.addf`/`arith.mulf`、`arith.cmpi`/`arith.cmpf`、`arith.select`，以及 `arith.extsi`/`arith.sitofp` 等位宽与数值转换。`arith` 方言借助操作数类型和逐元素语义天然支持标量、向量和张量重载。`arith.constant` 是 SSA 常量的标准来源，配合 builtin 的 `IntegerAttr`/`FloatAttr`。注意溢出处理与未定义行为（undefined behavior）：有符号溢出通过 `overflowflags` （`nsw`/`nuw`）等属性显式标注，把 C 语言 UB 提升到 IR 层面表达，与 [[#ub Dialect]] 协同。`arith` 不含超越函数，由 [[#math Dialect]] 承载，共同构成标量计算底座。

## async Dialect
`async` 方言是对*异步执行模型*的抽象，把任务的并发与依赖关系提到 IR 层面表达，用于在不绑定具体线程和运行时实现细节的前提下*描述非阻塞计算*。核心类型是 `!async.token` 标记一个无返回值异步操作的完成、`!async.value` 标记带返回结果的异步 handle、`!async.group` 表示一组并行结果。标志性 Op 是 `async.execute`，划定一段可异步调度的区域并产出 token/value；`async.await`，`async.await_all` 在需要结果处阻塞等待，显式同步；`async.yield` 返回区域结果。

`async`常作为 [[#scf Dialect]]、[[#linalg Dialect]] 等并行循环切分后的承接层，也与 [[#gpu Dialect]] 的异步流（stream）协同表达 host-device 之间的重叠执行。在 lowering 上，`async` 经由运行时库（Async Runtime）降到 [[#llvm Dialect]]，把抽象的 token/value 实现为协程或线程池调用，是 MLIR 表达*任务级并行（task-level parallelism）* 的统一中介。

## bufferization Dialect
`bufferization` 是对 Bufferization 这一过程的抽象——把*值语义/value semantics* 的 [[#tensor Dialect]] 转换为*引用语义/reference semantics* 的 [[#memref Dialect]]，为不可变的张量分配实际缓冲区，是 MLIR 从数学计算落到内存读写的关键一步。核心 Op 是 `bufferization.to_buffer`（tensor → memref）和`bufferization.to_tensor`（memref → tensor）二者标记值语义计算和引用语义内存读写的边界；`bufferization.alloc_tensor` 显式*物化/materialize* 一块张量 buffer；`bufferization.materialize_in_destination`，`bufferization.dealloc`管理落地（落到内存中）与内存释放。

`bufferization`的设计精髓是 One-Shot Bufferize：借助接口在*原地/in-place* 分析张量的读写与别名（甚至用到并查集），尽可能复用 buffer、避免冗余拷贝、确有冲突时插入复制，从而生成高效的内存访问方案。紧密配合 [[#linalg Dialect]] 等结构化操作 *Destination-Passing Style/目的地传递风格*（操作显式接受输出 buffer）并与 [[#tensor Dialect]]，[[#memref Dialect]] 三者构成 MLIR 张量编译的标准 lowering 路径。

## cf Dialect
`cf` 是对（非结构化）控制流的抽象，即传统编译器中基于*基本块/basic block* 与*分支/branch* 的 CFG 表示，处于结构化控制流被拍平之后，lowering 到底层 IR 之前的层次。核心 Op 有：`cf.br` 无条件跳转、`cf.cond_br` 有条件跳转、`cf.switch` 多路分支跳转；跳转目标为同一 Region 内的其他基本块，并通过*块参数/block arguments* 传递值，实现了 MLIR/LLVM 以快参数替代 Phi 函数的策略。

`cf` 与 [[#scf Dialect]] 鲜明对照：[[#scf Dialect]] 用 `scf.for` 和 `scf.if` 等 Op 保留循环与分支的结构信息，便于分析变换；而一旦下降到 `cf`，结构信息展开为块、分支与跳转，分析精度随之下降。`cf` 向下几乎一一对应 [[#llvm Dialect]] 的分支指令。

## complex Dialect
`complex` 方言是对*复数（complex number）算术*的抽象，补齐 [[#arith Dialect]] 与 [[#math Dialect]] 只覆盖实数标量的空白。它以 builtin 的 `ComplexType`（如 `complex<f32>`）为操作数类型，`complex.create` 由实部与虚部构造复数、`complex.re` / `complex.im` 取分量；运算上既有 `complex.add` / `complex.sub` / `complex.mul` / `complex.div` 等基本四则，也有 `complex.abs` / `complex.exp` / `complex.log` / `complex.sqrt` / `complex.conj` 等对应 [[#math Dialect]] 的复数版超越与共轭操作。

`complex` 服务于科学计算、信号处理（如 FFT）等天然需要复数的负载，在 lowering 时把复数运算拆解为实部/虚部上的实数运算，最终降到 [[#arith Dialect]] / [[#math Dialect]] 乃至 [[#llvm Dialect]]，与前二者共同构成 MLIR 的标量计算底座。

## linalg Dialect
`linalg` 是对*结构化操作 structured operations*的抽象，以统一的代数框架表达线性代数与张量计算，是 ML 编译中承上启下 lowering 的核心枢纽。`linalg` 的精髓在于一套*结构化语义*：每个操作携带*indexing maps* （一组 [[#affine Dialect]] 的 `AffineMap`，刻画各操作数如何由迭代维度索引）与*iterator types* （标记每个维度 parallel 还是 reduction），把**完美循环嵌套**结构编码进 Op 本身语义之中。`linalg` 实质上只有一个 `linalg.generic` Op，其他 `linalg.matmul`、`linalg.conv_2d`、`linalg.fill` 等 具名操作 named ops 则是其语法糖，可随时泛化回 `linalg.generic`。 `linalg` 天然采用 Destination-Passing Style，与 [[#bufferization Dialect]] 紧密配合，在张量和缓冲两种语义上都成立。围绕 `linalg` 的 分块 tiling、融合 fusion、向量化 vectorization 等变换是性能优化的主场，常借助 [[#Transform Dialect]] 调度，向下分发到 [[#affine Dialect]]、[[#scf Dialect]]、[[#vector Dialect]]，是 IREE, torch-mlir 等框架的中端基石。

`linalg` 可谓最重要的中端方言。

## math Dialect
`math` 方言是对*数学函数*的抽象，承载 [[#arith Dialect]] 之外的*超越函数（transcendental）* 与高级数学运算，二者共同构成 MLIR 的标量计算底座——[[#arith Dialect]] 管四则、比较、位运算等基本算子，`math` 则管 `math.exp` / `math.log` / `math.sin` / `math.cos` / `math.sqrt` / `math.rsqrt` / `math.pow` / `math.tanh` / `math.erf` 等需要复杂实现的函数，以及 `math.fma`、`math.absf`、`math.floor` / `math.ceil` 等。与 [[#arith Dialect]] 一致，其操作借操作数类型支持标量、向量与张量的*逐元素*语义。`math` 的特别之处在于*多样的下降策略*：同一函数既可降到 [[#llvm Dialect]] 的 intrinsic 或调用 libm，也可经 *math-to-funcs / polynomial-approximation* 等 pass 用多项式近似就地展开（利于向量化与 GPU 执行），并允许通过快速数学标志在精度与性能间取舍。复数版本的对应运算则归 [[#complex Dialect]]。

## memref Dialect
`memref` 是对*已分配内存的引用语义*的抽象，是 MLIR 的核心内存模型，与值语义的 [[#tensor Dialect]] 相对：`tensor` 是不可变的数学值，而 `memref` 是可读写、有地址、可别名的内存缓冲区引用。其类型 `memref<MxNxT, layout, memory_space>` 不仅记录形状与元素类型，还携带 layout map ，一个 [[#affine Dialect]] 的 `AffineMap`，描述逻辑下标到线性内存地址的映射，表达行列主序、步长 strided、子视图等布局；和 memory space， 区分 global、workgroup、private 等，对接 [[#gpu Dialect]] 的内存层次。核心 Op 有分配与释放 `memref.alloc` / `memref.alloca` / `memref.dealloc`、读写 `memref.load` / `memref.store`、以及不复制数据只改变视图的 `memref.subview` / `memref.view` / `memref.reshape` / `memref.collapse_shape` / `memref.expand_shape`。

`memref` 通常由 [[#tensor Dialect]] 经 [[#bufferization Dialect]] 转换得到，既可承接 [[#affine Dialect]] 的仿射访存，也是 [[#linalg Dialect]] 在 buffer 语义下的操作对象，其布局信息配合 [[#dlti Dialect]] 降为 [[#llvm Dialect]] 的指针、GEP 与 load/store，是连接张量计算与物理内存的关键。

## quant Dialect
`quant` 是对*量化（quantization）* 的抽象，主要贡献一套描述量化数值的*类型系统*，服务于推理过程中把浮点模型转为低精度定点型数据以节省内存、提高算力。核心是 `!quant.uniform` 类型——刻画*均匀量化*的映射关系，以 scale 和 zero point 把实数域 `real = scale x (stored - zero_point)` 编码到整数域存储，并可区分逐张量 per-tensor 量化与逐通道 per-channel/per-axis 量化（后者更细化，每个通道有自己的 scale 和 zero-point）；此外还有非均匀映射 `!quant.calibrated`。`quant` 没有大量算术 Op，而是提供 `quant.qcast`、`quant.dcast`、`quant.scast`等转换操作，标记*量化边界*。

量化类型多由 [[#Tensor Operator Set Architecture (TOSA) Dialect]] 等前端算子携带，经下降后这些量化类型被实化 realize 为 [[#arith Dialect]] 上的整数定点运算，因此 `quant` 是连接带量化语义的高层算子与整数算术执行的类型桥梁。

## scf Dialect
`scf` 方言即 *Structured Control Flow*，是对*结构化控制流*的抽象，以保留结构信息的*嵌套区域(region)* 形式表达循环与分支，与基于基本块和跳转的非结构化 [[#cf Dialect]] 相对——`scf` 显式保留「这是一个循环/条件」的语义，远比展平为跳转的 `cf` 利于分析与变换。核心 Op 有 `scf.for`(带可选*归纳变量*与边界的计数循环)、`scf.while`(条件循环，分 before/after 两区域)、`scf.if`(条件分支)、以及 `scf.parallel`(并行循环)和 `scf.forall`(更现代的、带显式映射与共享输出的并行循环)。其关键设计是*以值传递循环携带状态*:`scf.for` 通过 `iter_args` 把循环间传递的值显式化为区域参数与返回值(从而无须可变内存即可表达规约/累加)，由 `scf.yield` 在每次迭代末交回更新值——这与 SSA 形式天然契合。在下降链上，`scf` 居于 [[#affine Dialect]] 之下、[[#cf Dialect]] 之上:[[#affine Dialect]] 的仿射循环可降为更一般的 `scf`(边界不再受仿射约束，但分析精度随之下降)，`scf` 再经 *control-flow lowering* 展平为 [[#cf Dialect]] 的基本块与分支;它也能承接 [[#linalg Dialect]] tiling 后的循环结构，是 MLIR 循环表示承上启下的中枢。

## shape Dialect
`shape` 方言是对*张量形状(shape)本身*的抽象,把形状作为*一等公民*提取出来独立计算与推理,服务*动态形状(dynamic shape)* 场景——当维度在编译期未知(记为 `?`)时,程序仍需表达「形状如何由输入推导」「形状须满足何种约束」。其类型 `!shape.shape`(可含未知维度)与 `!shape.size`、`!shape.witness` 构成基础;操作上 `shape.shape_of` 取张量形状、`shape.broadcast` 计算广播后形状、`shape.dim` / `shape.rank` 查询维度、`shape.cstr_broadcastable` / `shape.cstr_eq` 生成可广播/相等的*约束(constraint)*,并以 `!shape.witness` 串联——满足约束方可继续,从而把形状校验显式编码进 IR。它与 [[#tensor Dialect]] 紧密配合(`tensor.dim` 等可与之互转),计算出的形状最终常实化为 [[#arith Dialect]] / [[#index Dialect]] 上对 `index` 的实际运算,是 ML 编译器处理动态 shape、做形状推断与传播的专用方言。

## shard Dialect
`shard` 方言是对*张量分片(sharding)与分布式 SPMD 计算*的*声明式*抽象(原名 `mesh`)。它代表与 [[#acc Dialect]] / [[#omp Dialect]] 命令式指令相对的声明式一侧:你不手写通信,而是*声明*张量如何切分到一张*设备网格(device mesh)* 上,由编译器*推导*出所需的跨设备通信。核心是 `shard.grid`(声明多维设备网格拓扑)与 `shard.sharding` / `shard.shard`(声明某张量的各维如何沿网格轴切分或复制);在此基础上 `shard.all_reduce` / `shard.all_gather` / `shard.all_to_all` / `shard.reduce_scatter` 等表达*集合通信*。它通常作用于 [[#linalg Dialect]] / [[#tensor Dialect]] 之上,把单设备逻辑程序标注为分布式 SPMD 程序,经下降把集合通信映射到 [[#mpi Dialect]] 等实际收发原语,是 MLIR 表达*数据级分布式并行*的声明式枢纽。

## sparse_tensor Dialect
`sparse_tensor` 方言是对*稀疏张量编译*的抽象,核心思想是把「张量的稀疏性」表示为一种*类型属性*而非手写数据结构——通过 `#sparse_tensor.encoding` 属性附着到 [[#tensor Dialect]] 类型上,声明各维是*稠密(dense)* 还是*压缩(compressed)*、存储次序、指针/索引位宽等(可表达 CSR、CSC、COO、BCSR 等多种格式)。开发者只需像写稠密张量一样书写 [[#linalg Dialect]] 的结构化操作,*sparsifier*(稀疏编译器,即原 *Sparse Compiler*)便依据 encoding 与操作的 indexing maps / iterator types *自动生成*只遍历非零元的稀疏循环嵌套与协同迭代(co-iteration)代码。`sparse_tensor.convert` 在稠密与稀疏、不同稀疏格式间转换,`sparse_tensor.values` / `sparse_tensor.positions` / `sparse_tensor.coordinates` 等访问底层存储。它把数十年稀疏代码生成的技巧自动化,下降时产出 [[#scf Dialect]] / [[#memref Dialect]] 上的实际稀疏遍历,是 MLIR 在稀疏线性代数方向的标志性贡献。

## tensor Dialect
`tensor` 方言是对*张量的值语义(value semantics)* 的抽象,是 MLIR 张量计算的核心类型载体,与引用语义的 [[#memref Dialect]] 相对——`tensor` 是*不可变的数学值*,没有地址、不可别名,对它的「修改」实为产出新值。其类型 `tensor<MxNxT>` 支持以 `?` 表达*动态维度*。核心 Op 多为*纯值操作*:`tensor.empty` 物化一个未初始化张量(作结构化操作的 destination),`tensor.extract` / `tensor.insert` 取/置单元素,`tensor.extract_slice` / `tensor.insert_slice` 处理子块(返回新张量),`tensor.collapse_shape` / `tensor.expand_shape` / `tensor.reshape` 改形,`tensor.pad` 填充,`tensor.dim` 查询维度(与 [[#shape Dialect]] 协同)。它是 [[#linalg Dialect]] 在 value 语义下的操作对象,经 [[#bufferization Dialect]] 转换为 [[#memref Dialect]] 落到实际内存,与 `memref`、`bufferization` 三者构成 MLIR 张量编译从「数学值」到「物理缓冲」的标准下降路径。

# 后端方言
## amdgpu Dialect
`amdgpu` 对接 AMD GCN/CDNA/RDNA 结构，暴露那些贴近 AMD 硬件、无法纳入抽象 [[#gpu Dialect]] 的、专属于 AMD 独有的（汇编）功能和 LLVM 内置功能。Lowering 链：[[#gpu Dialect]] 提供后端无关抽象，`amdgpu` 提供 AMD 硬件特性结构化封装，最终下降到 [[#rocdl Dialect]] （ROCDL/AMDGPU/LLVM  intrinsics）。`amdgpu` 的标志性能力是矩阵加速指令：`amdgpu.mfma`， Matrix Fused Multiply-Add 操作，对应 CDNA 的 Matrix Core，是 MI 系列加速 GEMM 和卷积的核心指令；`amdgpu.wmma`， Wave Matrix Multiply-Accumulate 操作。此外，它把 AMD 特有的内存模型显式表达：`amdgpu.raw_buffer_load`/`raw_buffer_store`/`raw_buffer_atomic_*` 通过*缓冲区描述符/buffer descriptor* 访存并带有硬件越界防护；`amdgpu.lds_barrier` 操作 AMD Local Data Share；`amdgpu.sched_barrier` 向后端调度器施加约束。`amdgpu` 是 ROCm 路径上 ML 算子获得 Matrix Core 性能的关键层，是 IREE、torch-mlir 等面向 AMD GPU 后端的必经方言。

## arm_neon Dialect
`arm_neon` 方言对接 *ARM NEON*（AArch64 的 *高级 SIMD / Advanced SIMD* 扩展），把那些无法由通用 [[#vector Dialect]] 方言充分表达的 NEON 专有指令暴露为可直接映射到底层 intrinsic 的 Op，在 lowering 链上位于 [[#vector Dialect]] 与 [[#llvm Dialect]]（ARM 后端 intrinsic）之间。它面向的是固定宽度（128 位）向量寄存器，提供 NEON 特有而通用向量运算难以泛化的操作，典型如 `arm_neon.intr.smmla`/`ummla`/`usmmla`——*整数矩阵乘累加（Matrix Multiply-Accumulate）* 指令，是 NEON 上加速低精度（int8）GEMM 与卷积、服务 ML 推理的关键。其设计定位与 [[#arm_sve Dialect]] （可伸缩向量）、[[#arm_sme Dialect]]（可伸缩矩阵）形成对照：`arm_neon` 是*固定长度*的传统 SIMD 路径，三者共同覆盖 ARM 平台从定长到可伸缩的向量化后端，是 MLIR 在 ARM CPU 上获得硬件 SIMD 性能的落地层。

## arm_sve Dialect
`arm_sve` 方言对接 *ARM SVE（Scalable Vector Extension，可伸缩向量扩展）*，暴露 SVE 专有而通用 [[#vector Dialect]] 方言无法表达的指令，在 lowering 链上位于 [[#vector Dialect]] 与 [[#llvm Dialect]]（ARM 后端 intrinsic）之间。它与 [[#arm_neon Dialect]] 的根本区别在于*向量长度不定（Vector-Length Agnostic， VLA）*——SVE 寄存器宽度由硬件实现决定（128 到 2048 位之间），同一份二进制无需重编译即可在不同宽度的实现上运行，因此其向量类型是*可伸缩（scalable）* 的，对应 MLIR 的 `vector<[n]xT>` 语法（方括号标记 scalable 维度）。配套的核心机制是*谓词（predicate）*：SVE 用掩码寄存器控制逐通道的激活与否，从而优雅处理无法被向量宽度整除的循环尾部，免去 [[#arm_neon Dialect]] 那种显式尾循环。标志性 Op 如 `arm_sve.smmla`/`ummla`（整数矩阵乘累加）等服务低精度 ML 计算。`arm_sve` 与定长的 [[#arm_neon Dialect]] 可伸缩矩阵的 [[#arm_sme Dialect]] 共同覆盖 ARM 向量后端，是 MLIR 面向新一代 ARM 服务器/HPC 芯片做可伸缩向量化的落地层。

## arm_sme Dialect
`arm_sme` 方言对接 *ARM SME（Scalable Matrix Extension，可伸缩矩阵扩展）*，是 ARM 向量后端从「向量」迈向「矩阵」的一层，在 lowering 链上位于 [[#vector Dialect]] 与 [[#llvm Dialect]]（ARM 后端 intrinsic）之间。它建立在 SVE 的*可伸缩（scalable）* 与*谓词（predicate）* 理念之上，但核心是一块二维的*片上累加寄存器 ZA（ZA storage / tile）*——SME 引入 *Streaming SVE 模式*，在该模式下以 ZA tile 为单位做*外积累加（outer product accumulate）*，单条指令即可完成矩阵块的乘累加，是定长 NEON、一维 SVE 之外专为 GEMM/卷积设计的矩阵加速路径。因此 `arm_sme` 比前两者多出对*二维可伸缩 tile* 的建模（同时在两个维度上可伸缩，类型形如 `vector<[n]x[n]xT>`），标志性操作围绕 ZA tile 的 load/store、outer product 与 tile 与向量之间的搬运展开。`arm_sme` 与定长的 `arm_neon`、一维可伸缩的 `arm_sve` 共同构成 MLIR 在 ARM 平台上「定长 SIMD → 可伸缩向量 → 可伸缩矩阵」的完整后端阶梯，面向最新 ARM 架构的高吞吐 ML 与 HPC 负载。

## emitc Dialect
`emitc` 方言是对生成 C / C++ 源代码这一目标的抽象，提供一条不经 [[#llvm Dialect]] 与 LLVM IR、而是直接发射 C 代码的后端出口服务于需要可读 C 源码、对接既有 C 工具链或面向缺乏 LLVM 后端的专用编译器（如某些 DSP / 加速器 SDK）的场景。它的 Op 与类型刻意贴合 C 语言构造：`emitc.call_opaque` 发射对任意 C 函数的调用、`emitc.literal` / `emitc.constant` 嵌入字面量、`emitc.include` 生成 `#include`、`emitc.for` / `emitc.if` 对应 C 的控制流语句，配合 `!emitc.opaque<"...">`（承载任意 C 类型字符串）、`!emitc.ptr`、`!emitc.lvalue` 等类型表达 C 的指针与左值语义。其特别之处在于 IR 不再是纯 SSA 的计算图，而带有命令式、有左值的 C 语义色彩。配套的 translation（`mlir-translate --mlir-to-cpp`）负责把 emitc IR 真正打印成 `.c` / `.cpp` 文本。

`emitc` 是各上层方言（如 [[#tosa Dialect]]、[[#linalg Dialect]] 经下降后）通向 C 源码而非机器码的标志性末端方言。

## gpu Dialect
`gpu` 是对*后端无关 GPU 编程模型*的抽象，以平台无关的方式建模 GPU 的并行执行与内存层次，向上承接 [[#scf Dialect]]、[[#linalg Dialect]] 等并行计算，向下统一分发到 [[#nvvm Dialect]]、[[#rocdl Dialect]]、[[#SPIR-V Dialect]] 等厂商后端，与硬件专有的 [[#nvgpu Dialect]]、[[#amdgpu Dialect]] 形成通用抽象-专有扩展的分工。`gpu` 建模了 GPU 的核心概念：`gpu.launch`、`gpu.launch_func` 启动 kernel 并指定 gird/block 维度；`gpu.func` 定义带 `gpu.kernel` 标记的设备端函数；`gpu.thread_id`、`gpu.block_id`、`gpu.block_dim` 等查询线程在网格中的索引；`gpu.barrier` 做块内同步；`gpu.shuffle` 表达 warp/wavefront 级数据交换；内存层次上区分 workgroup 与 private 地址空间。`gpu` 还以 `!gpu.async.token` 表达异步流，与 [[#async Dialect]] 协同 host-device 重叠执行。关键的 `gpu-kernel-outling` pass 把 `gpu.launch` 区域抽取为独立 `gpu.func` 模块，再经厂商 lowering 生成 PTX、AMDGCN、SPIR-V 汇编等。

`gpu` 是 MLIR 异构编译中承上启下的中枢方言。

## llvm Dialect
`llvm` 方言是对 *LLVM IR* 的*近乎一一对应的镜像*，在 MLIR 体系内重建 LLVM 指令、类型与属性，是绝大多数 lowering 路径的*最终汇聚层*——各方言降到此处后，经 *translation*（`mlir-translate --mlir-to-llvmir`）即可导出真正的 LLVM IR，接入 LLVM 后端生成各架构机器码。它在 MLIR 的语义约束下复刻 LLVM 构造：`llvm.func` / `llvm.call` / `llvm.return` 对应函数与调用，`llvm.add` / `llvm.icmp` / `llvm.getelementptr`（GEP）/ `llvm.load` / `llvm.store` / `llvm.br` / `llvm.cond_br` 对应运算、寻址与控制流；类型上有 `!llvm.ptr`（*不透明指针 / opaque pointer*）、`!llvm.struct`、`!llvm.array`、`!llvm.func` 等 LLVM 类型的 MLIR 表达。它仍保持 SSA 与基本块结构，与 [[#cf Dialect]] 的非结构化控制流自然衔接，并依赖 [[#dlti Dialect]] 提供的 data layout 做布局决策。

值得注意的是 `llvm` 方言不是 LLVM IR 的完全等价物，而是其在 MLIR 类型系统与可扩展机制下的投影；[[#nvvm Dialect]]、[[#rocdl Dialect]] 等厂商 intrinsic 方言均构建于其上，共同构成 MLIR 通往机器码的标准末端。

## mpi Dialect
`mpi` 是对*Message Passing Interface*的抽象，把 HPC 领域的事实标准的*分布式多进程通信模型* 暴露为 MLIR 的操作，服务跨节点的并行计算。它对应 MPI 的核心原语：`mpi.comm_world` 取得全局通信子，`mpi.comm_rank` / `mpi.comm_size` 查询当前进程的*秩(rank)* 与进程总数，`mpi.send` / `mpi.recv` 表达*点对点(point-to-point)* 通信，`mpi.allreduce` / `mpi.barrier` 等表达*集合(collective)* 通信，`mpi.init` / `mpi.finalize` 管理运行时生命周期;通信结果以 `mpi.retval` 等承载错误码与状态。它常作为 [[#shard Dialect]] 等张量分片/分布式抽象在跨进程层面的*下降目标*——上层以逻辑分片描述 SPMD 计算，经下降映射到具体的 MPI 收发与集合操作。在 lowering 上，`mpi` 经运行时绑定降到 [[#llvm Dialect]]，把方言操作实现为对 MPI 库(如 MPICH / Open MPI)函数的调用，是 MLIR 触及*多节点分布式 HPC* 场景的接口方言。

## nvgpu Dialect
`nvgpu` 方言对接 *NVIDIA* GPU，暴露那些过于贴近硬件、无法纳入后端无关的 [[#gpu Dialect]] 的 NVIDIA 专有能力，在 lowering 链上处于 [[#gpu Dialect]] 与 [[#nvvm Dialect]] 之间——`gpu` 提供平台中立抽象，`nvgpu` 提供 NVIDIA 硬件特性的*结构化封装*，最终再降到 `nvvm`。它聚焦于 Tensor Core 与异步内存等现代 NVIDIA 特性：`nvgpu.mma.sync` 表达 *warp 级矩阵乘累加(MMA)*、`nvgpu.ldmatrix` 为 MMA 加载寄存器分片，`nvgpu.device_async_copy` / `nvgpu.device_async_create_group` / `nvgpu.device_async_wait` 表达 Ampere 起的*异步拷贝(cp.async)*——把全局内存到共享内存的搬运与计算重叠;较新的 `nvgpu.tma.async.load` 等则对接 Hopper 的 *TMA(Tensor Memory Accelerator)* 与 `mbarrier`。它比 `gpu` 更专、比 `nvvm` 更抽象，是 ML 算子在 NVIDIA GPU 上榨取 Tensor Core 性能的关键中间层，被 [[#linalg Dialect]] 向 GPU 下降的路径所依赖。

## nvvm Dialect
`nvvm` 方言对接 *NVVM IR*——NVIDIA 基于 LLVM IR 扩展、面向 PTX 生成的中间层，是 MLIR 通往 NVIDIA GPU 的 intrinsic 末端方言，构建于 [[#llvm Dialect]] 之上并与之共同 translate 为带 NVVM intrinsic 的 LLVM IR，再由后端发射 *PTX* 汇编。它几乎一一对应 NVVM/PTX 的特殊操作：`nvvm.read.ptx.sreg.tid.x` 等读取线程/块的*特殊寄存器(special register)*、`nvvm.barrier0` 对应 `bar.sync` 块内同步、`nvvm.shfl.sync` 表达 *warp shuffle*、`nvvm.mma.sync` / `nvvm.wmma.*` 直接映射 Tensor Core 的 PTX 矩阵指令、`nvvm.cp.async.*` 对应异步拷贝指令。它处于 [[#nvgpu Dialect]] 与 [[#gpu Dialect]] 之下：上层的结构化 GPU 操作经下降到 `nvvm` 后即贴合 PTX 语义，是 NVIDIA 后端链 `gpu → nvgpu → nvvm → llvm` 的最底层方言，与 AMD 侧的 [[#rocdl Dialect]] 地位相当。

## ptr Dialect
`ptr` 方言是对*指针(pointer)* 这一概念的*方言无关抽象*，把原先与 [[#llvm Dialect]] 深度绑定的指针类型与寻址语义提炼出来，成为可被多个方言共享的独立底层设施，以减少对 `llvm` 方言的强依赖。其核心是 `!ptr.ptr` 类型——一个携带*地址空间(address space)* 的不透明指针，与 [[#memref Dialect]] 的高层、带形状与 layout 的内存引用形成对照:`memref` 是结构化的多维缓冲视图，`ptr` 则是更接近机器、仅表达「某地址空间中的一个地址」的底层句柄。配套操作围绕基本寻址与访存展开，如指针偏移计算(对应 LLVM 的 GEP 语义)、`ptr.load` / `ptr.store`、以及与整数/其他指针类型的转换;地址空间则借接口与 [[#dlti Dialect]] 的数据布局信息协同，决定指针位宽与对齐。

`ptr` 是 MLIR 推进*底层概念去 `llvm` 化、模块化*的一部分——让需要裸指针语义的方言无须各自重造或全盘依附 `llvm`，最终仍自然降到 [[#llvm Dialect]] 的 `!llvm.ptr`。

## rocdl Dialect
`rocdl` 方言即 *ROCm Device Library*，对接 AMD *ROCm* 软件栈面向 GPU 的设备端 intrinsic，是 MLIR 通往 AMD GPU 的*末端方言*，与 NVIDIA 侧的 [[#nvvm Dialect]] 地位相当。它构建于 [[#llvm Dialect]] 之上，与之共同 translate 为带 AMDGPU intrinsic 的 LLVM IR，再由 LLVM 的 AMDGPU 后端发射 *AMDGCN* 汇编。它几乎一一对应底层硬件操作:`rocdl.workitem.id.x` / `rocdl.workgroup.id.x` 等读取*工作项 / 工作组(work-item / workgroup)* 索引(对应 [[#nvvm Dialect]] 的 thread/block special register)，`rocdl.barrier` 做工作组同步，`rocdl.mfma.*` 直接映射 CDNA *Matrix Core* 的矩阵乘累加指令、`rocdl.wmma.*` 映射 RDNA 的矩阵指令，`rocdl.ds.*` 等对应 LDS 访问。

它处于 [[#amdgpu Dialect]] 与 [[#gpu Dialect]] 之下:上层结构化 GPU 操作经下降到 `rocdl` 后即贴合 AMDGCN 语义，构成 AMD 后端链 `gpu → amdgpu → rocdl → llvm` 的最底层，与 NVIDIA 链 `gpu → nvgpu → nvvm → llvm` 完全镜像对应。

## vcix Dialect
`vcix` 方言即 *Vector Coprocessor Interface eXtension*,对接 SiFive 为 RISC-V 向量扩展定义的*协处理器接口*,属后端生成类,服务于把自定义向量协处理器/加速器指令接入 RISC-V 向量编程模型。它暴露 VCIX 的特殊指令,使编译器能从 [[#vector Dialect]] 等上层向量表示下降到对接协处理器的 intrinsic,最终经 [[#llvm Dialect]] 由 RISC-V 后端发射相应指令。其定位与其他向量后端方言(如 ARM 的 [[#arm_sve Dialect]]、x86 的 [[#x86 Dialect]])平行,是 RISC-V 生态中触及厂商自定义向量加速硬件的接口方言,面向需要扩展指令的定制 RISC-V 处理器。


## vector Dialect
`vector` 方言是对*平台无关 SIMD 向量计算*的抽象,作为「上层结构化计算」与「各架构硬件向量指令」之间的统一中间层——上承 [[#linalg Dialect]] 等经*向量化(vectorization)* 产出的向量操作,下分发到 [[#arm_neon Dialect]] / [[#arm_sve Dialect]] / [[#x86 Dialect]] / [[#amx Dialect]] 等架构专有后端或 [[#llvm Dialect]] 的向量类型。它操作 builtin 的 `VectorType`(支持多维向量及 *scalable* 可伸缩维度)。标志性 Op 包括连接内存与寄存器的 `vector.transfer_read` / `vector.transfer_write`(带掩码与边界处理的高层搬运,是向量化的关键抽象)、表达矩阵乘累加语义的 `vector.contract`(携带 indexing maps 与 iterator types,与 [[#linalg Dialect]] 一脉相承)、规约 `vector.reduction` / `vector.multi_reduction`、数据重排 `vector.shuffle` / `vector.broadcast` / `vector.transpose` / `vector.extract` / `vector.insert`,以及形状调整 `vector.shape_cast`。它通过分级的 *vector lowering* 逐步把高层向量操作拆解为贴近硬件的形式,是 MLIR 在 CPU/GPU 上兑现 SIMD 性能的核心枢纽。

## wasmssa Dialect
`wasmssa` 方言即 *WebAssembly SSA*,对接 *WebAssembly(Wasm)*,属后端生成类,目标是把 WebAssembly 表示为符合 MLIR *SSA 形式*的方言——WebAssembly 原生是*基于栈(stack-based)* 的字节码,`wasmssa` 则将其重构为 SSA 值流形式,以便纳入 MLIR 的分析与变换框架。它建模 Wasm 的核心构造:数值运算、内存访问、函数与调用、以及结构化的控制流(Wasm 本身的 `block` / `loop` / `if` 是结构化的)等。该方言使 MLIR 能作为面向 WebAssembly 的编译目标或分析 Wasm 的工具基础,是 MLIR 生态向 Web/可移植字节码方向延伸的接口方言,目前属于较新的探索性后端方言。

## x86 Dialect
`x86` 方言对接 *Intel/AMD 的 x86 向量扩展*(*AVX / AVX2 / AVX-512* 等),暴露那些无法由平台无关 [[#vector Dialect]] 充分表达的 x86 专有 SIMD 指令,在 lowering 链上位于 [[#vector Dialect]] 与 [[#llvm Dialect]](x86 后端 intrinsic)之间。它把 x86 特有而通用向量运算难以泛化的操作呈现为可直接映射到底层 intrinsic 的 Op——典型如 AVX-512 的部分专用指令、`dot` 类点积、特殊重排与掩码操作等,使编译器在向量化后能选用 x86 上最高效的实现。其定位与 ARM 的 [[#arm_neon Dialect]] / [[#arm_sve Dialect]]、RISC-V 的 [[#vcix Dialect]] 平行,是 MLIR 在 x86 CPU 上榨取 SIMD 性能的后端落地层;更专门的 Intel 矩阵扩展 *AMX(Tile)* 则由独立的 amx 方言承担。

## xegpu Dialect
`xegpu` 方言对接 *Intel GPU*(Xe 架构),暴露其专有的、无法纳入后端无关 [[#gpu Dialect]] 的硬件能力,与 NVIDIA 的 [[#nvgpu Dialect]]、AMD 的 [[#amdgpu Dialect]] 在各自厂商路径中地位平行。它聚焦 Intel Xe 的矩阵加速与高效访存:以 `xegpu.create_nd_tdesc` 等创建*张量描述符(tensor descriptor)* 刻画带分块/布局的内存区域,`xegpu.load_nd` / `xegpu.store_nd` 借描述符做块状搬运,`xegpu.dpas`(*Dot Product Accumulate Systolic*)表达 Xe *systolic array* 上的矩阵乘累加(对接 Intel 的 *XMX / Xe Matrix eXtensions*),并有对应的预取等操作。它处于 [[#gpu Dialect]] 与 Intel GPU 后端(对接 [[#xevm Dialect]])之间,是 ML 算子在 Intel GPU 上获取矩阵引擎性能的关键中间层。

## xevm Dialect
`xevm` 方言对接 Intel GPU 的*底层 intrinsic 层*(Xe 架构,经由 Intel 的 LLVM 后端),是 Intel GPU 路径的*末端方言*,地位与 NVIDIA 的 [[#nvvm Dialect]]、AMD 的 [[#rocdl Dialect]] 相当。它构建于 [[#llvm Dialect]] 之上,把上层的 [[#xegpu Dialect]] 结构化操作下降为贴近 Xe 硬件的 intrinsic(包括矩阵指令、特殊寄存器访问与同步等),再经 translation 交由 Intel 的 GPU 编译后端生成最终设备代码。它使 Intel GPU 后端链补齐为 `gpu → xegpu → xevm → llvm`,与 NVIDIA 链(`gpu → nvgpu → nvvm → llvm`)、AMD 链(`gpu → amdgpu → rocdl → llvm`)三方完全镜像对称。

## SPIR-V Dialect
`spirv` 方言对接 *SPIR-V*——Khronos 为 Vulkan、OpenCL 等定义的标准*中间表示*,是 MLIR 中少见的、与目标 IR *近乎一一对应*的完整方言(类比 [[#llvm Dialect]] 之于 LLVM IR),目标是作为 GPU/异构计算的可移植编译出口。它完整复刻 SPIR-V 的结构:以 `spirv.module` 为容器并携带*版本、能力(capability)、扩展(extension)、内存模型*等;类型上有 `!spirv.array` / `!spirv.struct` / `!spirv.ptr`(带*存储类 storage class*)/ image/sampler 等;操作覆盖算术、`spirv.Load` / `spirv.Store`、控制流 `spirv.Branch` / `spirv.mlir.loop` / `spirv.mlir.selection`、函数 `spirv.func`,以及 *GLSL/OpenCL extended instruction set* 的数学函数等。它通常作为 [[#gpu Dialect]] 的下降目标之一(`gpu → spirv`),再经 *serialization* 直接产出二进制 SPIR-V,供 Vulkan/OpenCL 运行时消费,是 MLIR 通往 Khronos 生态、面向跨厂商 GPU 的标志性后端方言。