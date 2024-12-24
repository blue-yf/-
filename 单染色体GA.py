import numpy as np
import matplotlib.pyplot as plt
import random
import time


class Chromosome:
    def __init__(self, n):
        self.n = n
        self.priority_chromosome = np.zeros(n, dtype=int)

    def initialize_chromosomes(self):
        self.priority_chromosome = np.random.permutation(np.arange(1, self.n + 1))


class ProductionScheduler:
    def __init__(self, n, f, k, P, D_order, H_W, H_N, H_A, molds, X_alpha, d, chromosome, component_costs_map):
        self.n = n
        self.f = f
        self.k = k
        self.P = P
        self.D_order = D_order
        self.H_W = H_W
        self.H_N = H_N
        self.H_A = H_A
        self.molds = molds
        self.X_alpha = X_alpha
        self.d = d
        self.chromosome = chromosome
        self.component_costs_map = component_costs_map
        self.component_to_mold = {}
        for mold_type, component_ids in molds.items():
            for component_id in component_ids:
                self.component_to_mold[component_id] = mold_type

        # 解码染色体得到生产计划
        self.production_schedule, self.global_mold_type_schedule = self.decode_chromosome()

        # 初始化时间和状态矩阵
        self.S = np.zeros((f + 1, n + 1, k + 1), dtype=float)
        self.C = np.zeros((f + 1, n + 1, k + 1), dtype=float)
        self.W = np.zeros((f + 1, n + 1, k + 1), dtype=float)

        # 初始化每条流水线的当前时间
        self.current_time = np.zeros(f + 1, dtype=float)

        # 初始化模具使用状态
        self.mold_available = {mold: [] for mold in self.global_mold_type_schedule.keys()}

    def decode_chromosome(self):
        # 按优先级排序构件，并记录原始索引（即构件ID）
        sorted_components_with_indices = sorted(enumerate(self.chromosome.priority_chromosome), key=lambda x: x[1])
        sorted_component_indices = [i + 1 for i, _ in sorted_components_with_indices]  # 构件ID

        # 随机分配生产线
        line_assignments = {i: [] for i in range(1, self.f + 1)}
        for component_id in sorted_component_indices:
            assigned_line = random.randint(1, self.f)  # 随机选择 1 到 f 之间的生产线
            mold_type = next((mt for mt, cids in self.molds.items() if component_id in cids), None)
            if mold_type is None:
                raise ValueError(f"Component ID {component_id} not found in molds dictionary")

            line_assignments[assigned_line].append((component_id, mold_type))

        production_schedule = {i: [] for i in range(1, self.f + 1)}
        global_mold_type_schedule = {mt: [] for mt in self.molds.keys()}

        for line, components in line_assignments.items():
            production_schedule[line] = [component_id for component_id, mold_type in components]
            for component_id, mold_type in components:
                global_mold_type_schedule[mold_type].append(component_id)

        return production_schedule, global_mold_type_schedule

    def get_schedule_string(self):
        """
        将生产计划转换为字符串形式。
        """
        schedule_str = ""
        for line, components in self.production_schedule.items():
            schedule_str += f"Line {line}: "
            schedule_str += ", ".join(map(str, components))
            schedule_str += "\n"
        return schedule_str

    def check_constraints(self, i, j, h, T):
        # 约束1和2 (所有工序都需要检查)
        if h > 0 and T < self.C[i + 1, j + 1, h - 1]:
            return False
        if j > 0 and T < self.C[i + 1, j, h - 1]:
            return False

        # 约束3、4和5 (工作时间和加班时间约束)
        D = int(T // 24)
        if h == 3:
            if T <= 24 * D + self.H_W + self.H_A:
                pass
            elif T > 24 * D + self.H_W + self.H_A:
                return T >= 24 * (D + 1)
            else:
                return False
        elif h == 4:
            if T <= 24 * D + self.H_W:
                pass
            elif 24 * D + self.H_W < T < 24 * (D + 1):
                return T >= 24 * (D + 1)
            elif T > 24 * (D + 1):
                pass
            else:
                return False
        else:  # h not in [3, 4]
            if T <= 24 * D + self.H_W:
                pass
            elif T > 24 * D + self.H_W:
                return T + self.H_N >= 24 * (D + 1)
            else:
                return False

        # 约束 6 (模具约束 - 改进)
        component_id = self.production_schedule[i + 1][j - 1]
        mold_type = self.component_to_mold.get(component_id)
        if mold_type is not None:
            mold_usage_order = self.global_mold_type_schedule[mold_type]  # 直接使用global_mold_type_schedule
            l_alpha = mold_usage_order.index(component_id) + 1
            X_alpha = self.X_alpha.get(mold_type)
            if l_alpha > X_alpha:
                min_previous_completion_time = float('inf')
                for k in range(max(0, l_alpha - X_alpha), l_alpha - 1):
                    prev_comp_id = mold_usage_order[k]
                    for line_index in range(1, self.f + 1):
                        for comp_index in range(1, len(self.production_schedule[line_index]) + 1):
                            if self.production_schedule[line_index][comp_index - 1] == prev_comp_id:
                                min_previous_completion_time = min(min_previous_completion_time,
                                                                   self.C[line_index, comp_index, self.k])
                                break
                        else:
                            continue
                        break
                if T < min_previous_completion_time:
                    return False
        return True

    def schedule_production(self):
        self.current_time = np.zeros(self.f + 1, dtype=int)
        machine_available_times = np.zeros((self.f + 1, self.k + 1), dtype=float)

        for i in range(1, self.f + 1):
            line_schedule = self.production_schedule[i]
            num_components_on_line = len(line_schedule)
            for j in range(1, num_components_on_line + 1):
                component_id = line_schedule[j - 1]
                for h in range(1, self.k + 1):
                    # 获取该工序的可用时间，考虑机器和前序工序
                    T = max(machine_available_times[i, h], 0 if h == 1 else self.C[i, j, h - 1])
                    if j > 1:
                        T = max(T, self.C[i, j - 1, h])
                    while not self.check_constraints(i - 1, j - 1, h - 1, T):
                        T += 1.0

                    self.S[i, j, h] = T
                    if h in [3, 4]:
                        self.C[i, j, h] = T + self.P[j - 1, h - 1]
                    else:
                        self.C[i, j, h] = T
                    machine_available_times[i, h] = self.C[i, j, h]
                    if h == self.k:
                        mold_type = self.component_to_mold.get(component_id)
                        if mold_type is not None:
                            self.mold_available[mold_type].append(self.C[i, j, h])

        self.C_makespan = np.max(self.C[:, :, -1])
        self.component_completion_times = {}
        for i in range(1, self.f + 1):
            for j in range(1, len(self.production_schedule[i]) + 1):
                component_id = self.production_schedule[i][j - 1]
                completion_time = self.C[i, j, -1]
                if component_id not in self.component_completion_times or completion_time < completion_time:
                    self.component_completion_times[component_id] = completion_time

    def estimate_earliest_start_time(self, i, j, h):
        """
        估算满足所有约束条件的最早开始时间。
        """
        # 初始化为流水线的当前时间
        T = self.current_time[i + 1]  # 注意这里要加1

        # 工序约束（前一工序的完成时间）
        if h > 0:
            T = max(T, self.C[i + 1, j + 1, h])  # 注意这里要加1

        # 构件顺序约束（前一个构件同一工序的完成时间）
        if j > 0:
            T = max(T, self.C[i + 1, j, h + 1])  # 注意这里要加1

        # 模具约束
        component_id = self.production_schedule[i + 1][j]
        mold_type = next((mt for mt, cids in self.global_mold_type_schedule.items() if component_id in cids), None)
        if mold_type is not None and self.mold_available[mold_type]:
            last_release_time = max(self.mold_available[mold_type])
            T = max(T, last_release_time)

        # 加班时间和工作时间约束
        T = self.enforce_time_constraints(T, h)

        return T

    def enforce_time_constraints(self, T, h):
        """
        根据工序时间限制（正常工作时间、加班时间等）调整开始时间。
        """
        D = int(T // 24)  # 当前天数
        if h == 3:  # 第3道工序的特殊约束
            if T <= 24 * D + self.H_W + self.H_A:
                return T
            else:
                return 24 * (D + 1)
        elif h == 4:  # 第4道工序的特殊约束
            if T <= 24 * D + self.H_W:
                return T
            elif T < 24 * (D + 1):
                return 24 * (D + 1)
            else:
                return T
        else:  # 其他工序的通用约束
            if T <= 24 * D + self.H_W:
                return T
            else:
                return max(T, 24 * (D + 1) - self.H_N)

    def simulate_stacking(self):
        sorted_components = sorted(self.component_completion_times.items(), key=lambda x: x[1])

        stacks = []
        current_stack = []

        for component_id, _ in sorted_components:

            if len(current_stack) < self.d:
                current_stack.append(component_id)

            else:
                stacks.append(current_stack)

                current_stack = [component_id]

        if current_stack:
            stacks.append(current_stack)

        return stacks

    def calculate_total_moving_cost(self, stacks, component_order, component_costs_map):
        """
        计算按照指定顺序从堆栈中取出构件的总移动成本。
        Args:
            stacks: 一个列表，列表中的每个元素是一个堆栈（列表），表示构件的堆叠顺序。
            component_order: 一个列表，表示构件的出库顺序。
            component_costs_map: 一个字典，表示每个构件的移动成本。
        Returns:
            总移动成本。
        """
        total_cost = 0
        # 创建构件成本映射
        cost_map = {key: value for key, value in component_costs_map.items()}

        # 将每个堆栈复制一份，避免修改原始堆栈
        temp_stacks = [list(stack) for stack in stacks]

        for component_id in component_order:
            stack_to_remove_from = None
            for stack in temp_stacks:
                if component_id in stack:
                    stack_to_remove_from = stack
                    break
            if stack_to_remove_from is None:
                raise ValueError(
                    f"Component {component_id} not found in any stack. Check stacking logic or component order.")

            position_in_stack = stack_to_remove_from.index(component_id)

            # 计算要移动的构件的成本
            moving_cost = sum(
                cost_map[stack_to_remove_from[i]] for i in range(position_in_stack + 1, len(stack_to_remove_from)))
            total_cost += moving_cost

            # 从堆栈中移除构件
            stack_to_remove_from.pop(position_in_stack)

        return total_cost

    def calculate_objectives(self):
        # 在计算目标函数值之前，先解码染色体，生成最新的生产计划
        self.production_schedule, self.global_mold_type_schedule = self.decode_chromosome()

        # 然后调用schedule_production方法来安排生产并计算最大完工时间
        self.schedule_production()
        # 从schedule_production方法中，我们已经得到了self.C_makespan作为最大完工时间
        max_makespan = self.C_makespan

        # 调用simulate_stacking方法来获取堆叠的构件组
        stacks = self.simulate_stacking()  # 不再传递 self.C

        # 复制堆栈以避免后续修改影响搬运成本计算
        stacks_copy = [list(stack) for stack in stacks]

        # 最后，调用calculate_total_moving_cost方法来计算移动成本
        move_count = self.calculate_total_moving_cost(stacks_copy, self.D_order, self.component_costs_map)

        # 合并目标函数
        combined_objective = 0.5 * max_makespan + 0.5 * move_count  # 修改此处，添加权重

        # 保存目标函数值
        self.objectives = combined_objective

        # 返回目标函数值（虽然这里不直接返回，但可以在需要时访问self.objectives）
        return self.objectives


class GeneticAlgorithm:
    def __init__(self, population_size, max_generations, crossover_rate, mutation_rate, n, f, k, P, D_order, H_W, H_N,
                 H_A, molds, X_alpha, d, component_costs):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n = n
        self.f = f
        self.k = k
        self.P = P
        self.D_order = D_order
        self.H_W = H_W
        self.H_N = H_N
        self.H_A = H_A
        self.molds = molds
        self.X_alpha = X_alpha
        self.d = d
        self.component_costs = component_costs  # 保存 component_costs 以便在需要时使用
        # 创建一个从构件编号到成本值的映射
        self.component_costs_map = {i + 1: cost for i, cost in enumerate(component_costs)}

        random.seed(42)
        np.random.seed(42)

        # 初始化种群
        self.population = []
        for _ in range(population_size):
            chromosome = Chromosome(n)
            chromosome.initialize_chromosomes()
            # 创建 ProductionScheduler 实例时传入 component_costs_map
            individual = ProductionScheduler(n, f, k, P, D_order, H_W, H_N, H_A, molds, X_alpha, d, chromosome,
                                             self.component_costs_map)
            # 计算目标函数值并保存
            individual.production_schedule, individual.global_mold_type_schedule = individual.decode_chromosome()
            individual.calculate_objectives()
            self.population.append(individual)

    def evaluate_population(self):
        """
        返回当前种群中每个个体的适应度值列表。
        适应度值是目标值的倒数，因为我们要最小化目标值。
        """
        fitness = [1 / ind.objectives for ind in self.population]
        return fitness

    def selection_roulette(self, fitness):
        """
        使用轮盘赌选择，基于适应度值的概率选择个体。
        """
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        selected_index = np.random.choice(len(self.population), p=probabilities)
        return self.population[selected_index]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            start_point = random.randint(1, self.n - 2)
            end_point = random.randint(start_point + 1, self.n)

            # 只复制优先级染色体
            child1_priority = np.copy(parent1.chromosome.priority_chromosome)
            child2_priority = np.copy(parent2.chromosome.priority_chromosome)

            # 交换优先级染色体在交叉区域内的基因
            child1_priority[start_point:end_point], child2_priority[start_point:end_point] = \
                child2_priority[start_point:end_point], child1_priority[start_point:end_point]

            # 修复重复基因问题
            repaired_fragment1 = self.repair_chromosome_with_mapping(child1_priority,
                                                                     parent1.chromosome.priority_chromosome,
                                                                     parent2.chromosome.priority_chromosome,
                                                                     start_point, end_point)
            repaired_fragment2 = self.repair_chromosome_with_mapping(child2_priority,
                                                                     parent2.chromosome.priority_chromosome,
                                                                     parent1.chromosome.priority_chromosome,
                                                                     start_point, end_point)

            # 将修复后的片段放回到完整的优先级染色体中
            child1_priority[start_point:end_point] = repaired_fragment1
            child2_priority[start_point:end_point] = repaired_fragment2

            # 创建新的Chromosome对象，只初始化priority_chromosome
            child1_chromosome = Chromosome(self.n)
            child1_chromosome.priority_chromosome = child1_priority

            child2_chromosome = Chromosome(self.n)
            child2_chromosome.priority_chromosome = child2_priority

            # 创建子代个体并计算目标函数值
            child1 = ProductionScheduler(self.n, self.f, self.k, self.P, self.D_order, self.H_W, self.H_N, self.H_A,
                                         self.molds, self.X_alpha, self.d, child1_chromosome,
                                         self.component_costs_map)
            child1.calculate_objectives()

            child2 = ProductionScheduler(self.n, self.f, self.k, self.P, self.D_order, self.H_W, self.H_N, self.H_A,
                                         self.molds, self.X_alpha, self.d, child2_chromosome,
                                         self.component_costs_map)
            child2.calculate_objectives()

            return child1, child2
        else:
            return parent1, parent2

    def repair_chromosome_with_mapping(self, chromosome, parent1, parent2, start_point, end_point):
        # 建立映射关系
        mapping = {parent2[i]: parent1[i] for i in range(start_point, end_point)}

        # 初始化一个集合来跟踪已使用的基因
        used_genes = set(chromosome[start_point:end_point])

        # 初始化结果染色体片段
        repaired_fragment = np.copy(chromosome[start_point:end_point])

        # 遍历需要修复的部分中的每个基因
        for i in range(len(repaired_fragment)):
            if repaired_fragment[i] in mapping:
                mapped_gene = mapping[repaired_fragment[i]]
                if mapped_gene in used_genes:
                    # 找到一个未使用的基因进行替换
                    for j in range(1, self.n + 1):
                        if j not in used_genes:
                            used_genes.add(j)
                            repaired_fragment[i] = j
                            break
                else:
                    used_genes.add(mapped_gene)  # 添加映射后的基因到used_genes
                    repaired_fragment[i] = mapped_gene  # 将映射后的基因赋值给repaired_fragment[i]
            else:
                used_genes.add(repaired_fragment[i])

        return repaired_fragment

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            if self.n < 2:  # 如果构件数量小于2，则无法进行交换变异
                return individual  # 不进行变异，直接返回
            idx1, idx2 = random.sample(range(self.n), 2)
            individual.chromosome.priority_chromosome[idx1], individual.chromosome.priority_chromosome[idx2] = \
                individual.chromosome.priority_chromosome[idx2], individual.chromosome.priority_chromosome[idx1]
        return individual

    def get_individual(self, index):
        """根据索引获取种群中的个体（ProductionScheduler对象）。"""
        return self.population[index]

    def run(self):
        best_objectives_per_generation = []  # 记录每代的最优目标值

        # 初始化种群和适应度
        fitness = self.evaluate_population()

        for generation in range(self.max_generations):
            # 找到当前种群最优个体
            best_index = np.argmax(fitness)
            best_individual = self.get_individual(best_index)
            best_objective = best_individual.objectives  # 获取目标函数值
            best_objectives_per_generation.append(best_objective)

            print(f"Generation {generation + 1}: Best Objective = {best_objective:.4f}")

            offspring_population = []
            while len(offspring_population) < self.population_size:
                # 通过轮盘赌选择选择父代个体
                parent1 = self.selection_roulette(fitness)
                parent2 = self.selection_roulette(fitness)

                # 进行交叉操作
                child1, child2 = self.crossover(parent1, parent2)

                # 进行变异操作
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                # 将子代个体添加到子代种群中
                offspring_population.extend([child1, child2])

            # 计算子代种群的目标值
            for individual in offspring_population:
                individual.calculate_objectives()

            #  使用子代作为新种群
            self.population = offspring_population
            # 计算子代适应度值
            fitness = self.evaluate_population()

        # 输出最终结果
        best_index = np.argmin(best_objectives_per_generation)
        best_objective = best_objectives_per_generation[best_index]
        best_individual = self.get_individual(best_index)
        print("\nBest Solution Values:")
        print(f"{best_objective:.2f}")
        print("\nProduction Schedule for Selected Best Solution:")
        print(best_individual.get_schedule_string())

        return best_objectives_per_generation


# 初始化和运行代码
n, f, k, d = 10, 2, 6, 2
H_W, H_N, H_A = 8, 16, 4
X_alpha = {1: 2, 2: 2, 3: 2}
molds = {1: [1, 3, 4, 6, 9], 2: [2, 8], 3: [5, 7, 10]}
# 手动输入每个构件的6道工序的生产时间
P = np.array([
    [2.0, 1.6, 2.4, 12, 2.5, 1.0],  # 构件1的生产时间
    [3.4, 4.0, 4.0, 12, 2.4, 2.5],  # 构件2的生产时间
    [0.8, 1.0, 1.2, 12, 0.8, 1.7],  # 构件3的生产时间
    [0.6, 0.8, 1.0, 12, 0.6, 2.0],  # 构件4的生产时间
    [3.0, 3.6, 2.4, 12, 2.4, 3.0],  # 构件5的生产时间
    [3.0, 3.2, 3.0, 12, 3.0, 1.6],  # 构件6的生产时间
    [1.3, 0.9, 2.4, 12, 1.9, 1.8],  # 构件7的生产时间
    [1.7, 1.4, 1.1, 12, 0.9, 0.7],  # 构件8的生产时间
    [2.2, 1.8, 1.2, 12, 2.3, 0.7],  # 构件9的生产时间
    [1.6, 3.2, 2.3, 12, 2.1, 2.7]  # 构件10的生产时间
])
# 构件的交货顺序
D_order = [1, 3, 5, 6, 2, 9, 8, 4, 7, 10]

# 构件1到构件10的单位移动成本
component_costs = [1, 2, 4, 6, 5, 2, 8, 3, 2, 1]

ga = GeneticAlgorithm(population_size=100, max_generations=100, crossover_rate=0.9, mutation_rate=0.2, n=n, f=f, k=k,
                      P=P, D_order=D_order, H_W=H_W, H_N=H_N, H_A=H_A, molds=molds, X_alpha=X_alpha, d=d,
                      component_costs=component_costs)
# 记录开始时间
start_time = time.time()

# 运行GA算法并记录结果
best_objectives_per_generation  = ga.run()

# 记录结束时间
end_time = time.time()
execution_time = end_time - start_time

# 打印算法运行时间
print(f"Algorithm Execution Time: {execution_time:.2f} seconds")

# 绘制迭代图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(best_objectives_per_generation) + 1), best_objectives_per_generation)
plt.xlabel('Generation')
plt.ylabel('Best Objective ')
plt.grid(True)
plt.show()