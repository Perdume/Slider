package org.example.Learning;

import org.example.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class AIMoves {
    public List<List<String>> result = new ArrayList<>(); //Solve#2 List
    List<int[][]> resultboard = new ArrayList<>();
    List<int[]> ailoclist = new ArrayList<>();
    private List<String> removeRepeat(List<String> ls, String lm){
        if(lm == null){
            return ls;
        }
        if(Objects.equals(lm, "w")&&ls.contains("s")){
            ls.remove("s");
        }
        if(Objects.equals(lm, "s")&&ls.contains("w")){
            ls.remove("w");
        }
        if(Objects.equals(lm, "d")&&ls.contains("a")){
            ls.remove("a");
        }
        if(Objects.equals(lm, "a")&&ls.contains("d")){
            ls.remove("d");
        }
        return ls;
    }
    public void getAIMoves(int[][] board, int[] playerLoc, int[] ailoc, List<String> fturnmove, String bmove) {
        String moved = "";
        VirtualBoard vb = new VirtualBoard(board, playerLoc, ailoc);
        if (fturnmove.size() == 10) {
            return;
        }
        List<String> overstuffing = new ArrayList<>(fturnmove);
        for (String str : removeRepeat(vb.getCanAIMove(), bmove)) {
            fturnmove = new ArrayList<>(overstuffing);
            vb.resetBoard();
            int[] sol = null;
            if (Objects.equals(str, "w")) {
                sol = vb.AIVirtualMove("y", -1);
                fturnmove.add("w");
                moved = "w";
            } else if (Objects.equals(str, "s")) {
                sol = vb.AIVirtualMove("y", 1);
                fturnmove.add("s");
                moved = "s";
            } else if (Objects.equals(str, "a")) {
                sol = vb.AIVirtualMove("x", -1);
                fturnmove.add("a");
                moved = "a";
            } else if (Objects.equals(str, "d")) {
                sol = vb.AIVirtualMove("x", 1);
                fturnmove.add("d");
                moved = "d";
            }
            if (sol == null) {
                continue;
            }
            if (sol[2] == 1) { //ToPlayerTurn
                if (!result.contains(AStarPathfinding.findPath(util.createboard(vb.board), new Node(ailoc), new Node(vb.AILoc)))) {
                    resultboard.add(vb.getBoard());
                    result.add(fturnmove);
                    ailoclist.add(vb.getAILoc());
                }
            } else {
                getAIMoves(util.createboard(vb.getBoard()), util.newloc(vb.getPlayerLoc()), util.newloc(vb.getAILoc()), new ArrayList<>(fturnmove), moved);
            }
        }
    }
}
