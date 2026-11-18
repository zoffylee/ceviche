# ceviche

[English](README.md) | **简体中文**

支持 Automatic Differentiation（自动微分）的电磁仿真工具。本仓库提供论文 [Forward-Mode Differentiation of Maxwell's Equations](https://arxiv.org/abs/1908.10507) 的代码。

![ceviche](img/horizontal-color.png)

## ceviche 是什么？

`ceviche` 提供两种求解 Maxwell’s equations（麦克斯韦方程组）的电磁仿真方法。

- finite-difference frequency-domain（FDFD，频域有限差分法）

- finite-difference time-domain（FDTD，时域有限差分法）

两种求解器都使用 PyTorch 张量，支持 forward-mode 和 reverse-mode automatic differentiation（前向和反向自动微分）。在 CPU 上，FDFD 使用 PARDISO 进行双精度求解，使用 SciPy 进行单精度求解，并通过 Torch 实现隐式求导。在 CUDA 上，FDFD 使用 Torch 实现的 GMRES。FDTD 的全部计算都由 Torch 在所选设备上完成。

你可以编写代码求解电磁问题，再利用自动微分计算结果对输入参数的导数。

由此可以进行 gradient-based optimization（基于梯度的优化）和 sensitivity analysis（灵敏度分析），也可以把电磁求解器接入机器学习模型，省去手工推导导数的工作。

## 示例

[原版 NumPy/Autograd 教程](https://github.com/fancompute/workshop-invdesign)包含以下 Notebook 示例。在当前版本中使用这些示例时，需要用 Torch 运算改写目标函数。

1. [使用 ceviche 运行 FDFD 仿真](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/01_First_simulation.ipynb)
2. [mode converter（模式转换器）的逆向设计](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/02_Invdes_intro.ipynb)
3. [加入制造约束与器件参数化](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/03_Invdes_parameterization.ipynb)
4. [wavelength-division multiplexer（波分复用器）的逆向设计及进阶主题](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/04_Invdes_wdm_scheduling.ipynb)

`examples/*` 目录中也提供了示例。

## 安装

本地开发使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。在仓库根目录运行以下命令。

```sh
uv sync --locked
uv run python -c "import ceviche; print(ceviche.__version__)"
uv build
```

项目在 `pyproject.toml` 中声明依赖，在 `uv.lock` 中锁定版本，使用 `uv_build` 后端构建软件包。PyPI 和 Python 下载使用 `pyproject.toml` 中配置的 CERNET 镜像。

项目从 PyTorch 官方索引安装 PyTorch 2.14.1 的 CUDA 12.6 构建版本。在已测试的 Windows/Tesla P100 系统（驱动版本 539.41）上，eager loading（立即加载）会导致 CUDA 初始化失败。请在启动 Python 前启用 CUDA lazy loading（延迟加载）。

```powershell
$env:CUDA_MODULE_LOADING = "LAZY"
uv run python -c "import torch; print(torch.ones(1, device='cuda'))"
```

开发检查使用项目已锁定版本的 Ruff 和 ty。

```sh
uv run ruff format .
uv run ruff check .
uv run ty check
```

## 仿真与梯度

初始网格和源可以是 NumPy 数组或张量。场和 Jacobian（雅可比矩阵）均以 Torch 张量返回。可微目标函数应使用 Torch 运算。NumPy 输入默认为双精度，张量保留其数据类型和设备。创建设计张量时，可显式指定 `dtype=torch.float64`。

```python
import math

import torch

from ceviche import fdfd_ez

# Use "cuda" for the P100, after setting CUDA_MODULE_LOADING as shown above.
device = "cpu"
eps = torch.full((16, 12), 2.0, dtype=torch.float64, device=device, requires_grad=True)
source = torch.zeros_like(eps)
source[7, 5] = 1.0
simulation = fdfd_ez(2 * math.pi * 200e12, 80e-9, eps, [3, 3])
hx, hy, ez = simulation.solve(source)
loss = ez.abs().square().mean()
loss.backward()
print(eps.grad.norm().item())
```

`fdfd_hz` 返回 `(Ex, Ey, Hz)`。组装矩阵和恢复电场时，均使用 Yee 网格每条边上相邻两处相对介电常数的算术平均值。如果边上的平均值为零或非有限值，组装时会抛出 `ValueError`。有限非零的平均值直接用于计算，包括复数值。

对于只有一个网格点的轴，若 Bloch phase（布洛赫相位）为 `φ`，则前向差分为 `(exp(iφ) - 1) / dL`，后向差分为 `(1 - exp(-iφ)) / dL`。在周期边界条件下（`φ = 0`），两者均为零。

`fdfd_mf_ez` 的源和场形状均为 `(2*Nsb+1, Nx, Ny)`，调制数组的形状为 `(harmonics, Nx, Ny)`。PML（perfectly matched layer，完全匹配层）宽度和 Bloch 相位均通过构造函数指定。每个仿真实例的网格形状固定，给 `eps_r` 重新赋值时会保留设计变量的计算图。

`SolverOptions(method="gmres", rtol=1e-10, maxiter=2000, restart=80)` 用于配置迭代求解。CPU 上默认对 `float64`/`complex128` 使用 PARDISO，对 `float32`/`complex64` 使用 SciPy；CUDA 上默认使用 GMRES。多数仿真可直接使用默认设置。CPU 上的 PARDISO 通过 `py-mkl-pardiso` 调用，支持实数和复数非对称线性系统。该封装只执行双精度计算，单精度输入在其内部会提升为双精度。可用 `method="scipy"` 显式选择 SciPy 参考后端，或用 `method="pardiso"` 指定 CPU 双精度求解。各求解路径均保留输入张量的数据类型，支持隐式梯度和 Jacobian-vector product（JVP，雅可比向量积）。CPU 和 CUDA 都可使用以下三种迭代方法。

- `gmres` 是一般 CUDA FDFD 系统的默认方法，使用 Jacobi preconditioning（雅可比预条件）。
- `bicgstab` 适用于一般线性系统，内存开销较低，但对某些矩阵和右端项的组合可能发生数值中断。其隐式反向和前向导数方程使用 GMRES，以更可靠地处理稀疏右端项或以基向量为右端项的情形。
- `cg` 仅适用于 Hermitian positive-definite（厄米正定）矩阵。一般 FDFD 系统并不满足这一条件。

CPU 上的 CG 和 BiCGSTAB 使用 SciPy；CUDA 版本在 Torch 中运行，并使用 Jacobi 预条件。也可通过 `solve_linear(A, b, iterative_method="bicgstab")` 调用。每次求解都会检查 true residual（真实残差），检查失败时抛出错误。`maxiter` 限制迭代次数，`restart` 控制 GMRES 的基向量个数。求解过程不会隐式将 CUDA 数据移至 CPU，也不会将大型稀疏矩阵展开为稠密矩阵。稀疏 FDFD 求解支持一阶反向梯度和 JVP，不支持二阶导数或通过 `vmap` 批量求解。单精度计算可能需要放宽容差。

通过 `jacobian(fun, mode="reverse" | "forward" | "numerical")` 求得的矩阵以展平后的输出分量为行、输入分量为列。复数坐标的实部和虚部交错排列，每个复数元素占用两个实数坐标。对于标量损失，优先使用 `torch.autograd.grad` 或 `loss.backward()` 直接求梯度，无需构造完整的 Jacobian。

`fdtd(eps, dL, [px, py, pz])` 支持正实数介电常数和实数电流源。`forward(Jz=source)` 推进一个时间步，并返回包含场分量的字典。`run({"Jz": sources}, checkpoint_steps=32)` 接收形状为 `(steps, Nx, Ny, Nz)` 的源时间序列，返回最终场。Checkpointing（检查点重计算）在反向求导时按段重新计算完整状态，以减少需要保存的中间量。计算前向 JVP 时使用 `checkpoint_steps=0`。`run` 默认重置仿真，设置 `reset=False` 可从当前状态继续。调用 `initialize_fields()` 或给 `eps_r` 赋值会开始新的仿真轨迹。反向传播前应保持源张量的内容不变。在同一轨迹上多次调用反向传播时，遵循 PyTorch 的计算图生命周期规则。

`adam_optimize` 使用 Torch 原生 Adam 优化器。设为 `jac=None` 时，从目标函数自动求梯度；传入可调用对象时，由该对象提供梯度；设为 `jac=True` 时，目标函数需返回 `(loss, gradient)`。指定上下界后，每次更新都将参数投影到允许范围内。回调函数在更新前接收 `(iteration, loss_history, detached_parameters)`。

`modes.get_modes` 和 `modes.insert_mode` 都通过 SciPy 在 CPU 上执行不可微的预处理，即使其余仿真在 CUDA 上运行也是如此。输入的介电常数必须已与计算图分离（detached），返回张量位于输入设备上。

`get_modes(eps_cross, omega, dL, npml)` 默认返回 `(eps_eff, profiles)`。`profiles` 的每一列是一个单位范数的模式，其本征值为 `eps_eff = (beta / k0)**2`，其中 `k0 = omega / C_0`。在 effective-index model（EIM，有效折射率模型）中，可直接将此本征值用作相对介电常数。

```python
from ceviche.modes import get_modes

eps_eff, profiles = get_modes(eps_cross, omega, dL, npml=0)
eps_region = eps_eff[0]
```

需要有效折射率 `n_eff = beta / k0` 时，传入 `return_neff=True`。`insert_mode` 返回已写入所选模式场分布的源网格。

`viz.real`、`viz.abs` 和 `utils.imarr` 会将张量从计算图中分离，并传至 CPU 以便绘图。与其他外部工具交换数据时，使用 `tensor.to_numpy` 显式转换。

## 示例与测试

`examples/` 中的脚本和四个 Notebook 均使用 Torch API。Density filter（密度滤波器）位于 `ceviche.filters`，无需安装 scikit-image 或 Angler。运行 Notebook 时，以 `examples/` 为工作目录，以便读取随示例提供的数据。完整优化和时域仿真的计算开销可能较大，试跑时可减少迭代次数或时间步数。

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run python -m compileall -q ceviche tests examples
uv run python -m unittest discover -s tests -v
uv build
```

测试覆盖复数稀疏运算的导数、实数和复数材料及源的梯度、PML/Bloch 边界条件、多频耦合、具有解析解的离散平面波、FDTD checkpointing、优化器、滤波器、模式和 FFT（快速傅里叶变换）。CUDA 可用时会运行相应的对比测试。在已测试的 Windows/P100 系统上，请先按上文说明设置环境变量。迁移期间使用的临时检查脚本和参考数据未纳入仓库。

### 电磁基准测试

CPU 基准测试采用 1.55 µm 真空波长。每条断言都指出所检查的场、求解器或物理量。

| 基准 | 参考值与验收判据 |
| --- | --- |
| 均匀介质中的 FDTD 平面脉冲 | 两个探针测得的传播时延与 `n Δx / c` 的偏差在 2% 以内，两个探针处波形的 L2 范数相差不超过 1%。 |
| 正入射 PML | FDFD 在面电流源两侧分别拟合入射波与反射波。反射功率和外边界场强度均须低于各自入射值的百万分之一。FDTD 脉冲的剩余能量须低于周期边界对照值的百万分之一。 |
| 直平板波导 | 芯层宽 0.48 µm、折射率为 2，包层为空气，以偶对称 TE（横电）模的色散关系为参考。通过投影 FDFD 场提取纵向传播常数。在 80、40、20 nm 三种网格上，每次步长减半后的误差至多为前一次的 35%，最细网格的相对误差须低于 0.3%。[平板波导色散关系](https://utw10817.utweb.utexas.edu/2012/AmirSPIEPW2011.pdf) |
| 功率平衡与吸收 | 入射通量应等于出射通量与介质损耗积分之和，相对容差为 `1e-8`。折射率分别为 1.5 和 1.5 − 0.04i 时，功率透射率与 Beer–Lambert 定律预测值的绝对偏差均须在 0.005 以内。 |
| FDFD 精度与后端 | 在 64 × 48 周期网格上，使用三个具有解析解的离散 Fourier 模式为所有求解器构造同一参考场。相对 L2 误差上限为：单精度 SciPy `2e-5`，双精度 SciPy 和 PARDISO `1e-10`，单精度 GMRES `4e-5`。可选 CUDA GMRES 基准还会验证双精度，其误差上限为 `1e-9`。 |

单独运行 CPU 基准测试可使用以下命令。

```sh
uv run python -m unittest discover -s tests -p test_em_benchmarks.py -k TestCPUPhysics -v
```

在 P100 上，通过 PowerShell 运行可选的 CUDA 基准测试。

```powershell
$env:CUDA_MODULE_LOADING = "LAZY"
$env:CEVICHE_TEST_CUDA = "1"
uv run python -m unittest discover -s tests -p test_em_benchmarks.py -k TestCUDAPhysics -v
```

## 致谢

如果在研究或工作中使用了本项目，请引用以下论文。

    @article{hughes2019forward,
      title={Forward-Mode Differentiation of Maxwell’s Equations},
      author={Hughes, Tyler W and Williamson, Ian AD and Minkov, Momchil and Fan, Shanhui},
      journal={ACS Photonics},
      volume={6},
      number={11},
      pages={3010--3016},
      year={2019},
      publisher={ACS Publications}
    }

项目标志由 [@nagilmer](http://nadinegilmer.com/) 设计。
