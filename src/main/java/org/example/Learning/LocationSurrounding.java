package org.example.Learning;

import org.example.util;

import java.util.*;

public class LocationSurrounding {

    public List<List<String>> result = new ArrayList<>(); //Solve#2 List
    List<int[][]> resultboard = new ArrayList<>();
    List<int[]> enemyloclist = new ArrayList<>();
    int[] playerloc;

    public List<List<String>> markVisited(int[][] list, int[] loc, int[] enemyloc, Set<List<Integer>> invlocs) {
        List<List<String>> movements = new ArrayList<>();
        VirtualBoard vb = new VirtualBoard(util.createboard(list), util.newloc(loc), util.newloc(enemyloc));
        // Check if the position is within the bounds of the array
        int[][] moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        String[] moveSymbols = {"w", "s", "a", "d"};

        for (int i = 0; i < moves.length; i++) {
            int[] move = {loc[0] + moves[i][0], loc[1] + moves[i][1]};

            if (canMove(move, enemyloc) && !invlocs.contains(Arrays.asList(move[0], move[1]))) {
                if (list[move[0]][move[1]] == 0) {
                    List<String> movement = new ArrayList<>();
                    switch (moveSymbols[i]) {
                        case "w" -> vb.VirtualMove("y", -1);
                        case "s" -> vb.VirtualMove("y", 1);
                        case "a" -> vb.VirtualMove("x", -1);
                        case "d" -> vb.VirtualMove("x", 1);
                    }
                    movement.add(moveSymbols[i]);
                    movements.add(movement);
                    resultboard.add(vb.board);
                    enemyloclist.add(vb.enemyLoc);
                    playerloc = loc;
                    result.add(movement);
                } else {
                    invlocs.add(Arrays.asList(loc[0], loc[1]));
                    List<List<String>> subMovements = markVisited(list, move, enemyloc, invlocs);
                    for (List<String> subMovement : subMovements) {
                        subMovement.add(0, moveSymbols[i]);
                        movements.add(subMovement);
                    }
                }
            }
        }

        if(resultboard.size() != enemyloclist.size()){
            System.out.println("Unmatch");
        }
        if(result.size() != enemyloclist.size()){
            System.out.println("Unmatch");
        }

        return movements;
    }

    private boolean canMove(int[] loc, int[] enemyloc) {
        int row = loc[0];
        int col = loc[1];
        return row >= 0 && row <= 6 && col >= 0 && col <= 6 && (col != enemyloc[1] || row != enemyloc[0]);
    }
}

