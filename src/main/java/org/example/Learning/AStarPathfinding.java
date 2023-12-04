package org.example.Learning;


import java.util.*;

class Node {
    int x, y, value, g, h;
    Node parent;

    public Node(int x, int y) {
        this.x = x;
        this.y = y;
        this.g = 0;
        this.h = 0;
        this.parent = null;
    }
    public Node(int[] loc) {
        this.x = loc[0];
        this.y = loc[1];
        this.g = 0;
        this.h = 0;
        this.parent = null;
    }

    public double getF() {
        return g + h;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Node node = (Node) obj;
        return x == node.x && y == node.y;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y);
    }
}

class Edge {
    Node start, end;
    double cost;

    public Edge(Node start, Node end, double cost) {
        this.start = start;
        this.end = end;
        this.cost = cost;
    }
}

public class AStarPathfinding {
    public static List<Node> findPath(int[][] grid, Node start, Node goal) {
        Set<Edge> edges = generateEdges(grid);
        PriorityQueue<Node> openSet = new PriorityQueue<>(Comparator.comparingDouble(node -> node.getF()));
        Set<Node> closedSet = new HashSet<>();

        start.h = (int) heuristic(start, goal);
        openSet.add(start);

        while (!openSet.isEmpty()) {
            Node current = openSet.poll();

            if (current.equals(goal)) {
                return reconstructPath(current);
            }

            closedSet.add(current);

            for (Edge edge : edges) {
                if (edge.start.equals(current) && !closedSet.contains(edge.end)) {
                    double tentativeG = current.g + edge.cost;

                    if (!openSet.contains(edge.end) || tentativeG < edge.end.g) {
                        edge.end.parent = current;
                        edge.end.g = (int) tentativeG;
                        edge.end.h = (int) heuristic(edge.end, goal);

                        if (!openSet.contains(edge.end)) {
                            openSet.add(edge.end);
                        }
                    }
                }
            }
        }

        return Collections.emptyList();
    }

    private static Set<Edge> generateEdges(int[][] grid) {
        Set<Edge> edges = new HashSet<>();

        for (int x = 0; x < grid.length; x++) {
            for (int y = 0; y < grid[0].length; y++) {
                if (grid[x][y] != 0) {
                    Node current = new Node(x, y);

                    if (x > 0 && grid[x - 1][y] != 0) {
                        edges.add(new Edge(current, new Node(x - 1, y), 1));
                    }
                    if (x < grid.length - 1 && grid[x + 1][y] != 0) {
                        edges.add(new Edge(current, new Node(x + 1, y), 1));
                    }
                    if (y > 0 && grid[x][y - 1] != 0) {
                        edges.add(new Edge(current, new Node(x, y - 1), 1));
                    }
                    if (y < grid[0].length - 1 && grid[x][y + 1] != 0) {
                        edges.add(new Edge(current, new Node(x, y + 1), 1));
                    }
                }
            }
        }

        return edges;
    }

    private static double heuristic(Node a, Node b) {
        // 유클리드 거리를 사용한 간단한 휴리스틱 함수
        return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
    }

    private static List<Node> reconstructPath(Node goal) {
        List<Node> path = new ArrayList<>();
        Node current = goal;

        while (current != null) {
            path.add(current);
            current = current.parent;
        }

        Collections.reverse(path);
        return path;
    }

    public List<String> findpath(int[][] grid, int[] goalpos, int[] startpos) {

        Node start = new Node(startpos[0], startpos[1]); // 시작점 (x, y, value)
        Node goal = new Node(goalpos[0], goalpos[1]);  // 목적지 (x, y, value)

        List<Node> path = findPath(grid, start, goal);

        return convertPathToString(path);
    }
    private static List<String> convertPathToString(List<Node> path) {
        List<String> pathString = new ArrayList<>();

        for (int i = 1; i < path.size(); i++) {
            Node current = path.get(i);
            Node previous = path.get(i - 1);

            if (current.x > previous.x) {
                pathString.add("D");
            } else if (current.x < previous.x) {
                pathString.add("A");
            } else if (current.y > previous.y) {
                pathString.add("S");
            } else if (current.y < previous.y) {
                pathString.add("W");
            }
        }

        return pathString;
    }
}
