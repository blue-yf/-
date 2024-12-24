import numpy as np
import matplotlib.pyplot as plt
import random
import time


class Chromosome:
    def __init__(self, n, f):
        self.n = n
        self.f = f
        self.priority_chromosome = np.zeros(n, dtype=int)
        self.line_chromosome = np.zeros(n, dtype=int)

    def initialize_chromosomes(self):
        self.priority_chromosome = np.random.permutation(np.arange(1, self.n + 1))
        self.line_chromosome = np.random.randint(1, self.f + 1, self.n)


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
        # self.component_costs = component_costs  # 注释掉，这个属性没用到
        self.component_costs_map = component_costs_map  # 保存 component_costs_map

        # 解码染色体得到生产计划和模具类型计划
        self.production_schedule, self.global_mold_type_schedule = self.decode_chromosome()

        # 初始化开始时间、完成时间和等待时间矩阵，从1开始索引
        self.S = np.zeros((f + 1, n + 1, k + 1), dtype=float)
        self.C = np.zeros((f + 1, n + 1, k + 1), dtype=float)
        self.W = np.zeros((f + 1, n + 1, k + 1), dtype=float)

        # 初始化每条流水线的当前时间
        self.current_time = np.zeros(f + 1, dtype=int)

        # 初始化模具使用状态
        self.mold_available = {mold: [] for mold in self.global_mold_type_schedule.keys()}

    def decode_chromosome(self):
        line_assignments = {i: [] for i in range(1, self.f + 1)}
        for position in range(self.n):
            component_id = position + 1
            assigned_line = self.chromosome.line_chromosome[position]

            # 添加验证以检查生产线编号是否有效
            if assigned_line < 1 or assigned_line > self.f:
                print(f"Error in decode_chromosome: Invalid assigned line {assigned_line} for component {component_id}")
                raise ValueError(f"Invalid assigned line {assigned_line} for component {component_id}")

            priority = self.chromosome.priority_chromosome[position]
            mold_type = next((mt for mt, cids in self.molds.items() if component_id in cids), None)
            if mold_type is None:
                raise ValueError(f"Component ID {component_id} not found in molds dictionary")

            line_assignments[assigned_line].append((priority, component_id, mold_type))

        production_schedule = {i: [] for i in range(1, self.f + 1)}
        global_mold_type_schedule = {mt: [] for mt in self.molds.keys()}

        for line, components in line_assignments.items():
            sorted_components = sorted(components, key=lambda x: x[0])
            production_schedule[line] = [comp[1] for comp in sorted_components]
            for priority, component_id, mold_type in sorted_components:
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
        # 检查约束条件1和2
        if h > 0 and T < self.C[i + 1, j + 1, h]:  # 注意这里要加1
            return False
        if j > 0 and T < self.C[i + 1, j, h + 1]:  # 注意这里要加1
            return False

        # 检查约束条件3（第3道工序）
        if h == 3:
            D = int(T / 24)
            if T <= 24 * D + self.H_W + self.H_A:
                pass  # 满足条件
            elif T > 24 * D + self.H_W + self.H_A:
                return T >= 24 * (D + 1)
            else:
                return False

        # 检查约束条件4（第4道工序）
        if h == 4:
            D = int(T / 24)
            if T <= 24 * D + self.H_W:
                pass  # 满足条件
            elif 24 * D + self.H_W < T < 24 * (D + 1):
                return T >= 24 * (D + 1)
            elif T > 24 * (D + 1):
                pass  # 满足条件
            else:
                return False

        # 检查约束条件5（除第3、4道工序外的其他工序）
        if h not in [3, 4]:
            D = int(T / 24)
            if T <= 24 * D + self.H_W:
                pass  # 满足条件
            elif T > 24 * D + self.H_W:
                return T + self.H_N >= 24 * (D + 1)
            else:
                return False

        # 检查约束条件6（模具约束）
        component_id = self.production_schedule[i + 1][j]
        mold_type = next((mt for mt, cids in self.global_mold_type_schedule.items() if component_id in cids),
                         None)
        if mold_type is not None and mold_type in self.mold_available:
            last_release_time = max(self.mold_available[mold_type]) if self.mold_available[mold_type] else 0
            if T < last_release_time:
                # 模具不可用，增加等待时间
                return False

        # 所有约束条件都满足
        return True

    def schedule_production(self):
        self.current_time = np.zeros(self.f + 1, dtype=int)

        for i in range(1, self.f + 1):  # 从1开始
            line_schedule = self.production_schedule[i]
            num_components_on_line = len(line_schedule)
            for j in range(1, num_components_on_line + 1):  # 从1开始
                component_id = line_schedule[j - 1]  # 注意这里要减1，因为list是从0开始的
                T = 0.0
                for h in range(1, self.k + 1):  # 从1开始
                    while not self.check_constraints(i - 1, j - 1, h - 1, T):  # 注意这里要减1
                        T += 1.0

                    # 计算开始时间 S
                    self.S[i, j, h] = T

                    # 计算完成时间 C，等于开始时间加上该工序的生产时间
                    self.C[i, j, h] = T + self.P[j - 1, h - 1]  # 注意这里要减1来获取正确的索引

                    # 更新当前流水线的当前时间
                    self.current_time[i] = max(self.current_time[i], self.C[i, j, h])

                    # 如果这是最后一个工序，更新模具的可用时间
                    if h == self.k:
                        mold_type = next(
                            (mt for mt, cids in self.global_mold_type_schedule.items() if component_id in cids), None)
                        if mold_type is not None:
                            self.mold_available[mold_type].append(self.C[i, j, h])

        # 计算最大完工时间
        self.C_makespan = np.max(self.C[:, :, -1])

        for i in range(1, self.f + 1):
            for j in range(1, len(self.production_schedule[i]) + 1):
                component_id = self.production_schedule[i][j - 1]

        self.component_completion_times = {}
        for i in range(1, self.f + 1):
            for j in range(1, len(self.production_schedule[i]) + 1):
                component_id = self.production_schedule[i][j - 1]
                completion_time = self.C[i, j, -1]
                if component_id not in self.component_completion_times or completion_time > \
                        self.component_completion_times[component_id]:
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
        # 首先调用schedule_production方法来安排生产并计算最大完工时间
        self.schedule_production()
        # 从schedule_production方法中，我们已经得到了self.C_makespan作为最大完工时间
        max_makespan = self.C_makespan

        # 调用simulate_stacking方法来获取堆叠的构件组
        stacks = self.simulate_stacking()  # 不再传递 self.C

        # 复制堆栈以避免后续修改影响搬运成本计算
        stacks_copy = [list(stack) for stack in stacks]

        # 最后，调用calculate_total_moving_cost方法来计算移动成本
        move_count = self.calculate_total_moving_cost(stacks_copy, self.D_order, self.component_costs_map)

        # 保存目标函数值
        self.objectives = (max_makespan, move_count)

        # 返回目标函数值（虽然这里不直接返回，但可以在需要时访问self.objectives）
        return self.objectives




class NSGAII:
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
        np.random.seed(42)  # 如果NumPy的随机数生成器也被使用的话

        # 初始化最佳调度方案和栈堆划分列表
        self.best_schedules = []
        self.best_stacks = []

        # 初始化种群
        self.population = []
        for _ in range(population_size):
            chromosome = Chromosome(n, f)
            chromosome.initialize_chromosomes()
            # 创建 ProductionScheduler 实例时传入 component_costs_map
            individual = ProductionScheduler(n, f, k, P, D_order, H_W, H_N, H_A, molds, X_alpha, d, chromosome,
                                             self.component_costs_map)
            # 计算目标函数值并保存
            individual.calculate_objectives()
            self.population.append(individual)

    def evaluate_population(self):
        """
        返回当前种群中每个个体的目标函数值列表。
        """
        objectives = [ind.objectives for ind in self.population]
        return objectives

    def non_dominated_sort(self, population, objectives):
        fronts = [[]]
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        ranks = [0] * len(population)

        for p in range(len(population)):
            for q in range(len(population)):
                if self.dominates(objectives[p], objectives[q]):
                    dominated_solutions[p].append(q)
                elif self.dominates(objectives[q], objectives[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            Q = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)

        return fronts, ranks

    def dominates(self, obj1, obj2):
        return all(x <= y for x, y in zip(obj1, obj2)) and any(x < y for x, y in zip(obj1, obj2))

    def calculate_crowding_distance(self, front, objectives):
        """
        计算种群中某个前沿（Pareto front）的拥挤距离。
        如果 front 为空，直接返回空列表。
        如果前沿中只有一个或两个个体，为它们分配一个默认的拥挤距离值。
        """
        if not front:
            return []

        n_obj = len(objectives[0])
        distances = [0] * len(front)

        if len(front) <= 2:
            # 如果前沿中只有一个或两个个体，为它们分配一个较大的默认拥挤距离值
            distances = [float('inf')] * len(front)
        else:
            for i in range(n_obj):
                front_objectives = np.array([objectives[idx][i] for idx in front])

                # 如果所有个体的目标值都相同，则跳过该目标的拥挤距离计算
                if np.all(front_objectives == front_objectives[0]):
                    continue

                # 检查分母是否为零，并避免除以零
                if front_objectives[-1] == front_objectives[0]:
                    continue

                sorted_indices = np.argsort(front_objectives)
                distances[sorted_indices[0]] = float('inf')  # 边界点的距离设为无穷大
                distances[sorted_indices[-1]] = float('inf')

                for j in range(1, len(front) - 1):
                    idx = sorted_indices[j]
                    # 避免除以零
                    if front_objectives[-1] != front_objectives[0]:
                        distances[idx] += (front_objectives[j + 1] - front_objectives[j - 1]) / (
                                front_objectives[-1] - front_objectives[0])

            # 移除任何 nan 值
            distances = [float('inf') if np.isnan(d) else d for d in distances]

        return distances

    def selection_tournament(self, population, fronts, objectives):
        front1_idx, front2_idx = random.sample(range(len(fronts)), 2)
        front1, front2 = fronts[front1_idx], fronts[front2_idx]

        if not front1:
            idx1 = random.choice(range(len(population)))
        else:
            idx1 = random.choice(front1)

        if not front2:
            idx2 = random.choice(range(len(population)))
        else:
            idx2 = random.choice(front2)

        obj1, obj2 = objectives[idx1], objectives[idx2]
        if self.dominates(obj1, obj2) or (
        self.crowding_distances[front1_idx][idx1] > self.crowding_distances[front2_idx][
            idx2] if front1_idx == front2_idx else True):
            return population[idx1]
        else:
            return population[idx2]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            start_point = random.randint(1, self.n - 2)
            end_point = random.randint(start_point + 1, self.n)

            # 复制父代染色体属性到新对象中
            child1_priority = np.copy(parent1.chromosome.priority_chromosome)
            child1_line = np.copy(parent1.chromosome.line_chromosome)
            child2_priority = np.copy(parent2.chromosome.priority_chromosome)
            child2_line = np.copy(parent2.chromosome.line_chromosome)

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

            # 创建新的Chromosome对象
            child1_chromosome = Chromosome(self.n, self.f)
            child1_chromosome.priority_chromosome = child1_priority
            child1_chromosome.line_chromosome = child1_line

            child2_chromosome = Chromosome(self.n, self.f)
            child2_chromosome.priority_chromosome = child2_priority
            child2_chromosome.line_chromosome = child2_line

            # 创建子代个体并计算目标函数值
            child1 = ProductionScheduler(self.n, self.f, self.k, self.P, self.D_order, self.H_W, self.H_N, self.H_A,
                                         self.molds, self.X_alpha, self.d, child1_chromosome,
                                         self.component_costs_map)  # 注意这里传递了 component_costs_map
            child1.calculate_objectives()  # 计算子代个体的目标函数值

            child2 = ProductionScheduler(self.n, self.f, self.k, self.P, self.D_order, self.H_W, self.H_N, self.H_A,
                                         self.molds, self.X_alpha, self.d, child2_chromosome,
                                         self.component_costs_map)  # 注意这里传递了 component_costs_map
            child2.calculate_objectives()  # 计算子代个体的目标函数值

            return child1, child2
        else:
            return parent1, parent2

    def repair_chromosome_with_mapping(self, chromosome, parent1, parent2, start_point, end_point):
        # 建立映射关系
        mapping = {parent2[i]: parent1[i] for i in range(start_point, end_point)}

        # 初始化一个集合来跟踪已使用的基因
        used_genes = set(chromosome[start_point:end_point])

        # 初始化结果染色体片段（只复制需要修复的部分）
        repaired_fragment = np.copy(chromosome[start_point:end_point])

        # 遍历需要修复的部分中的每个基因
        for i in range(len(repaired_fragment)):
            if repaired_fragment[i] in mapping:
                # 如果当前基因在映射中，获取其映射后的基因
                mapped_gene = mapping[repaired_fragment[i]]

                # 如果映射后的基因已在结果片段中使用，则需要替换
                if mapped_gene in used_genes:
                    # 找到一个未使用的基因进行替换
                    for j in range(1, self.n + 1):
                        if j not in used_genes:
                            used_genes.add(j)
                            repaired_fragment[i] = j
                            break
            else:
                # 如果当前基因不在映射中，直接添加到已使用基因的集合中
                used_genes.add(repaired_fragment[i])

        # 返回修复后的染色体片段（只返回需要修复的部分）
        return repaired_fragment

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.n), 2)

            # 只交换优先级染色体的值
            individual.chromosome.priority_chromosome[idx1], individual.chromosome.priority_chromosome[idx2] = \
                individual.chromosome.priority_chromosome[idx2], individual.chromosome.priority_chromosome[idx1]

        return individual

    def get_individual(self, index):
        """根据索引获取种群中的个体（ProductionScheduler对象）。"""
        return self.population[index]

    def run(self):
        last_optimal_objectives = []  # 添加此行记录最优前沿

        for generation in range(self.max_generations):
            # 评价当前种群的目标函数值
            objectives = self.evaluate_population()

            # 进行非支配排序并计算拥挤距离
            self.fronts, self.ranks = self.non_dominated_sort(self.population, objectives)
            self.crowding_distances = [self.calculate_crowding_distance(front, objectives) if front else [] for
                                       front in
                                       self.fronts]

            # 输出当前迭代的最优帕累托前沿
            optimal_pareto_front = self.fronts[0]
            optimal_objectives = [objectives[idx] for idx in optimal_pareto_front]

            # 这里使用列表覆盖，保证只有最后一代的最优解被保存
            last_optimal_objectives = optimal_objectives


            print(f"Generation {generation + 1}: Optimal Pareto Front Values:")
            for obj in optimal_objectives:
                print(f"  Makespan: {obj[0]:.2f}, Move Count: {obj[1]:.2f}")

            # 创建子代种群
            offspring_population = []
            while len(offspring_population) < self.population_size:
                # 通过锦标赛选择选择父代个体
                parent1 = self.selection_tournament(self.population, self.fronts, objectives)
                parent2 = self.selection_tournament(self.population, self.fronts, objectives)

                # 进行交叉操作
                child1, child2 = self.crossover(parent1, parent2)

                # 进行变异操作
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                # 将子代个体添加到子代种群中
                offspring_population.extend([child1, child2])

            # 对子代种群中的每个个体调用calculate_objectives
            for individual in offspring_population:
                individual.calculate_objectives()

            # 合并父代和子代种群
            combined_population = self.population + offspring_population
            combined_objectives = objectives + [ind.objectives for ind in offspring_population]

            # 对合并后的种群进行非支配排序和拥挤距离计算
            self.fronts, self.ranks = self.non_dominated_sort(combined_population, combined_objectives)
            self.crowding_distances = [self.calculate_crowding_distance(front, combined_objectives) if front else []
                                       for
                                       front in self.fronts]

            # 从合并后的种群中选择新的父代种群
            new_population = []
            i = 0
            while len(new_population) < self.population_size:
                front = self.fronts[i]
                if front:  # 检查前沿是否非空
                    crowding_distances_for_front = self.crowding_distances[i]
                    # 检查拥挤距离列表是否非空且长度与前沿长度匹配
                    if crowding_distances_for_front and len(crowding_distances_for_front) == len(front):
                        # 根据拥挤距离和非支配等级选择个体
                        to_select = min(len(front), self.population_size - len(new_population))
                        sorted_front = sorted(front,
                                              key=lambda x: (
                                                  self.ranks[x], -crowding_distances_for_front[front.index(x)]))
                        new_population.extend(combined_population[x] for x in sorted_front[:to_select])
                    else:
                        # 如果拥挤距离列表为空或长度与前沿长度不匹配，则打印错误消息并随机选择个体
                        print(
                            f"Error: Crowding distance list is empty or length ({len(crowding_distances_for_front)}) does not match front length ({len(front)}) for front {i}")
                        to_select = min(len(front), self.population_size - len(new_population))
                        new_population.extend(random.sample(front, to_select))
                i += 1

            # 更新父代种群
            self.population = new_population

        self.best_pareto_front = last_optimal_objectives  # 此处进行替换

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

nsga2 = NSGAII(population_size=100, max_generations=100, crossover_rate=0.9, mutation_rate=0.1, n=n, f=f, k=k,
               P=P, D_order=D_order, H_W=H_W, H_N=H_N, H_A=H_A, molds=molds, X_alpha=X_alpha, d=d,
               component_costs=component_costs)
# 记录开始时间
start_time = time.time()

# 运行NSGA-II算法并记录结果
best_objectives_per_generation = nsga2.run()

# 记录结束时间
end_time = time.time()
execution_time = end_time - start_time

# 打印算法运行时间
print(f"Algorithm Execution Time: {execution_time:.2f} seconds")

# 绘制最优前沿图
plt.figure(figsize=(10, 8))

# 获取最后一代的最优帕累托前沿和对应的目标函数值
if nsga2.best_pareto_front:
    optimal_objectives = nsga2.best_pareto_front

    # 打印最优帕累托前沿点的目标值
    print("\nOptimal Pareto Front of Last Generation Values:")
    for obj in optimal_objectives:
        print(f"  Makespan: {obj[0]:.2f}, Move Count: {obj[1]:.2f}")

    # 绘制最优帕累托前沿
    plt.scatter([obj[0] for obj in optimal_objectives], [obj[1] for obj in optimal_objectives],
                label='Optimal Pareto Front')

    plt.xlabel('Max Makespan')
    plt.ylabel('Total Move Count')
    plt.title('Optimal Pareto Front of Last Generation')  # 添加标题
    plt.legend()
    plt.grid(False)
    plt.show()

    # 目标函数权重
    weight_makespan = 0.5
    weight_move_count = 0.5

    # 归一化目标函数值
    makespan_values = np.array([obj[0] for obj in optimal_objectives])
    move_count_values = np.array([obj[1] for obj in optimal_objectives])

    if np.max(makespan_values) == np.min(makespan_values):
        normalized_makespan = np.zeros_like(makespan_values)
    else:
        normalized_makespan = (makespan_values - np.min(makespan_values)) / (
                np.max(makespan_values) - np.min(makespan_values))

    if np.max(move_count_values) == np.min(move_count_values):
        normalized_move_count = np.zeros_like(move_count_values)
    else:
        normalized_move_count = (move_count_values - np.min(move_count_values)) / (
                np.max(move_count_values) - np.min(move_count_values))

    # 计算加权和，并找到最小化加权和的索引
    weighted_sum = weight_makespan * normalized_makespan + weight_move_count * normalized_move_count
    best_index = np.argmin(weighted_sum)
    best_solution = optimal_objectives[best_index]

    # 输出选择的最优解
    print("\nSelected Best Solution (Based on Equal Weights):")
    print(f"  Makespan: {best_solution[0]:.2f}, Move Count: {best_solution[1]:.2f}")

    # 解码最优解
    best_individual = nsga2.get_individual(nsga2.fronts[0][best_index])
    print("\nProduction Schedule for Selected Best Solution:")
    print(best_individual.get_schedule_string())

else:
    print("\nNo Pareto front found in the last generation.")
