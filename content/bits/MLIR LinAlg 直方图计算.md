---
tags: [ai-compiler]
created: 2026-04-29
---

- 我们有一个输入数组 `X`，每个元素是 0~9 的整数
- 我们想统计每个值出现的次数 → 直方图 `H[10]`
- 特点：
    - 每个输入元素会 **归约到对应的 bin**
    - 数据依赖（不同线程可能同时写 H[bin]）
    - 如果 frontend 确认支持 atomic 操作，可以安全 parallel

```mlir
#map_x = affine_map<(i) -> (i)>           // 输入数组索引
#map_h = affine_map<(i) -> (i)>           // 输出直方图索引（bin）

func.func @histogram(%X: tensor<1024xi32>,
                     %H: tensor<10xi32>) -> tensor<10xi32> {

  %result = linalg.generic {
      indexing_maps = [#map_x, #map_h],
      iterator_types = ["parallel", "reduction"],  // x parallel, H reduction
  } ins(%X : tensor<1024xi32>)
    outs(%H : tensor<10xi32>) {
    ^bb0(%x_elem: i32, %h_elem: i32):
      // 假设 x_elem 的值就是 bin 索引
      // 使用 atomic add 来避免并行冲突
      %sum = atomic.addi %h_elem, 1 : i32
      linalg.yield %sum : i32
  }

  return %result : tensor<10xi32>
}
```

- **Data-dependent reduction**：
    - 归约的目标 bin 由输入元素值 `%x_elem` 决定 → 数据依赖
    - 编译器需要知道如何安全处理这种动态索引的归约
- **Frontend/用户保证**：
    - 如果 frontend 保证底层硬件提供 atomic add，可以指定 **parallel semantics**
    - 编译器就可以把所有输入元素 parallel 执行，而不是串行累加
    - 此例在 [[MLIR LinAlg Dialect Documents]] Property 3 中被用作迭代器类型显式声明的典型场景。
- **优势**：
    - 避免串行循环 → 利用 GPU/CPU 并行
    - 安全归约 → 保证结果正确

