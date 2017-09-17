# Author: Yuxi Zhang
# Purpose: Solving a special case (Ladder) in Steiner Tree Problems using Genetic Algorithm
import random
import KruskalTree
import copy
import pygame

class LadderState(object):

    def __init__(self, n, terminals):
        # a set of n original terminals
        self.n = n
        self.terminals = terminals

        # create a list of (n-2) random points (hope converge to Steiner Points)
        self.sp = []
        for i in range(n*2-2):  # since 2 by n ladder, so the number of original points is 2n
            # append a tuple (x,y), representing Steiner Point
            self.sp.append((random.uniform(0,n-1), random.random()))

    def __lt__(self, other):
        return self.sp < other.sp

    # -----------------Methods & Class for Simulated Annealing ------------

    def getMST(self):
        # use Kruskal's algorithm to get Minimal Spanning tree over points in terminal set and sp lists
        # then 1/total length

        points = list(self.terminals) + self.sp

        # construct MST spanning over all points (set)
        KT = KruskalTree.KruskalTree(points)
        MST = KT.KruskalMST()
        return MST
    def treeLength(self):
        MST = self.getMST()

        return MST.length()

    def randomMove(self):
        # return a random move for creating new neighbor
        i = random.randrange(len(self.sp)) # choose which Steiner point to move
        (x,y) = (random.uniform(0,self.n-1), random.random()) # move the point to where
        return (i,x,y)

    def neighbor(self,move):
        neighbor = copy.deepcopy(self)
        (i,x,y) = move
        neighbor.sp[i] = (x,y)
        return neighbor

    def setGraph(self,state):
        # an environment that display and update the ladder using pygame
        pygame.init()
        win = pygame.display.set_mode((800, 800))
        # win.fill((0,0,0))

        ladder_width = win.get_width() * 3 // 4
        scalar = ladder_width // (state.n - 1)  # which is also the side of each unit grid
        start_x = win.get_width() // 2 - ladder_width // 2
        start_y = win.get_height() // 2 + scalar // 2
        scaledP = []
        for t in list(state.terminals) + state.sp:

            scaledP.append((start_x + t[0] * scalar, start_y - t[1] * scalar))


        def update(neighbor, mst, toChange):
            nonlocal win, scalar, start_x, start_y, scaledP
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            line_color = (0, 255, 0) if toChange else (255, 0, 0)

            for i in range(len(neighbor.sp)):
                scaledP[len(neighbor.terminals) + i] = (
                start_x + neighbor.sp[i][0] * scalar, start_y - neighbor.sp[i][1] * scalar)

            # display
            win.fill((0, 0, 0))

            for edge in mst.edges:
                pygame.draw.line(win, line_color, scaledP[edge[0]], scaledP[edge[1]], 3)
            # draw termials
            for i in range(len(neighbor.terminals)):
                pygame.draw.circle(win, (255, 255, 255), scaledP[i], scalar // 50)
            pygame.display.update()
            pygame.time.delay(5)

        return update

    def anneal(self,state, visualize):
        # use simulated annealing to improve the state
        if visualize == True:

            update = self.setGraph(state)

        temp = 10.0
        while temp > 0.01:
            temp *= 0.999
            neighbor = state.neighbor(state.randomMove())
            mst = neighbor.getMST()
            change = state.treeLength()-mst.length()
            toChange = change>0 or random.random() < math.exp(change/temp)
            if toChange:
                state = neighbor

            if (visualize):
                update(neighbor, mst, toChange)

        if visualize:
            while pygame.QUIT not in [e.type for e in pygame.event.get()]:
                pass    # wait for user to quit

        return state

    # ------------- M e t h o d s   &   C l a s s   for   G e n e t i c    A l g o r i t h m---------------
    def fitness(self):
        return 1/self.treeLength()

    def crossover(self,other):
        # pick a random index as the crossover point
        index = random.randrange(len(self.sp))
        sublist1 = self.sp[0:index]
        sublist2 = self.sp[index:]
        offspring_sp = sublist1+sublist2
        offspring = LadderState(self.n, self.terminals)
        offspring.sp = offspring_sp
        return offspring

    # Giving each potential Steiner Point a small probability of getting randomly changed
    def mutate(self):

        probability = random.uniform(0.01,0.02)
        for i in range(len(self.sp)):
            if random.random()<probability:
                mutated_x = random.uniform(0,self.n-1)
                mutated_y = random.random()
                self.sp[i] = (mutated_x,mutated_y)


class Population(object):

    def __init__(self):
        self.states = list()
        self.total_fitness = 0

    def add(self, state):
        fitness = state.fitness()
        self.total_fitness += fitness
        self.states.append((fitness,state))

    def best(self):
        (fitness, state) = max(self.states)
        return state

    def select(self):
        random_num = random.uniform(0,self.total_fitness)
        for state_tuple in self.states:
            random_num -= state_tuple[0]
            if random_num < 0:
                return state_tuple[1]

    def nextgen(self):
        nextGen = Population()
        elite = self.best()
        nextGen.add(elite)

        for count in range(len(self.states)-1):
            parent1 = self.select()
            parent2 = self.select()
            offspring = parent1.crossover(parent2)
            offspring.mutate()
            nextGen.add(offspring)
        return nextGen



# ------Main Program------
if __name__ == "__main__":
    import math

    def init(n):
        # initialize the terminal set and benchmark
        print("\nTest on 2 by ",n,"ladder terminal set")

        # BENCHMARK to compare with: from graph theory formula for ladders
        # L_n= √(〖(n(1+√3/2)-1)〗^2+1/4), n is odd
        #    =n(1+√3/2)-1,   n is even
        best_from_formula = n*(1+(3**0.5)/2)-1 if n%2 == 0 else math.sqrt((n*(1+(3**0.5)/2)-1)**2+1/4)
        print("According to formula, |SMT| = ", best_from_formula)

        # since terminal set for all the states when n is given is fixed, create outside and pass into constructor
        terminals = set()
        for i in range(n):
            terminals.add((i, 0))
            terminals.add((i, 1))

        return [terminals, best_from_formula]

    def ladderAnnealTest(n, visualize=False):
        '''
        :param n: number of terminals each row
        :param visualize: whether to animate
        :return: None
        '''
        init_result = init(n)
        terminals = init_result[0]
        shortest_len = init_result[1]

        print("Applying Simulated Annealing")
        state = LadderState(n,terminals)
        print("Initial tree length: ", state.treeLength())
        result = state.anneal(state, visualize)
        result_treelen = result.treeLength()
        print("Final tree length: ", result_treelen)

        # the length of the tree got from simulated annealing is how much longer than the length of the shortest tree
        # in theory
        print("Final Tree is ", (result_treelen-shortest_len)*100/shortest_len, "% longer than the shortest tree in theory.")

    def ladderGATest(n,pop,generations, visualize=False):
        init_result = init(n)
        terminals = init_result[0]
        shortest_len = init_result[1]

        # print test configuration
        print("Appyling Genetic Algorithm \n",
              "Population size: ",pop,"\n",
              "Generations: ",generations)



        # Initialize population (random states)
        population = Population()
        for count in range(pop):
            population.add(LadderState(n,terminals))

        best = population.best()
        if visualize:
            update = best.setGraph(best)
        print(1/best.fitness(), "after 0 generations")


        # Evolve 100 generations
        generation = 0
        while generation<generations:
            generation += 1
            population = population.nextgen()
            best_candidate = population.best()
            candidate_mst = best_candidate.getMST()
            toChange = 1/(candidate_mst.length()) > best.fitness()
            if toChange:
                best = best_candidate
                print("optimized length to", 1/best.fitness(), "after ", generation, "generations")
            if visualize:
                update(best_candidate, candidate_mst, toChange)
            # TODOl better way

        # after all generations, compare with shortest length and print the result
        print("Final Tree is ", (best.treeLength()-shortest_len)*100/shortest_len, "% longer than the shortest tree in theory.")

        if visualize:
            while pygame.QUIT not in [e.type for e in pygame.event.get()]:
                pass  # wait for user to quit
# ------test-------------------
    for i in range(2,10):
        #at least 2
        #ladderGATest(5,100,200)
        ladderAnnealTest(i, True)
