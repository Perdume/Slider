package org.example.Learning;

import org.example.Move;

import java.util.*;

public class Calculate {
    int[][] CalBoard;
    int[] AILoc;
    int[] PlayerLoc;

    int[][] SolBoard;

    double score = 0;

    public Calculate(int[][] board, int[] plloc, int[] ailoc) {
        CalBoard = board;
        PlayerLoc = plloc;
        AILoc = ailoc;
    }

    public List<String> findsolve() {
        LocationSurrounding ls = new LocationSurrounding();
        ls.markVisited(CalBoard, AILoc, PlayerLoc, new HashSet<>());
        ls.Clearing();
        backtrackingAI ba = new backtrackingAI(ls.resultboard, PlayerLoc, ls.enemyloclist, ls.result, 1, 5);
        ba.cal();
        score = ba.score;
        System.out.println(Arrays.toString(ba.getoptimizedailoc()));
        System.out.println(Arrays.toString(ba.getoptimizedplayerloc()));
//        listtoscreen(ba.getoptimizedboard(), ba.getoptimizedplayerloc(), ba.getoptimizedailoc());
        List<String> ret = ba.getoptimizedmove();
        System.out.println(ret);
        return ret;
    }
    public double getScore(){
        return score;
    }
}
