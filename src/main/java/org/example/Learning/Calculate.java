package org.example.Learning;

import org.example.Move;

import java.util.*;

public class Calculate {
    int[][] CalBoard;
    int[] AILoc;
    int[] PlayerLoc;

    double score = 0;

    List<List<String>> solve = new ArrayList<>(); //Solve#1 List
    List<Integer> solveturn = new ArrayList<>();

    private final int[] goalLocation = new int[]{3, 3};

    public Calculate(int[][] board, int[] plloc, int[] ailoc) {
        CalBoard = board;
        PlayerLoc = plloc;
        AILoc = ailoc;
    }

    public List<String> findsolve() {
        AIMoves am = new AIMoves();
        am.getAIMoves(CalBoard, PlayerLoc, AILoc, new ArrayList<>(), null);
        backtrackingAI ba = new backtrackingAI(am.resultboard, PlayerLoc, am.ailoclist, am.result, 1, 3);
        ba.cal();
        score = ba.score;
        List<String> ret = ba.getoptimizedmove();
        return ret;
    }
    public double getScore(){
        return score;
    }
}
