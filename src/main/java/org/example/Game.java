package org.example;



import org.example.Learning.Calculate;

import java.util.*;

public class Game {

    private int[][] list;

    private int[] playerLocation = new int[]{0, 6};
    private int[] enemyLocation = new int[]{6, 0};
    private final int[] goalLocation = new int[]{3, 3};

    Boolean winnerset = false;

    public void start(){
        resetboard();
        playerTurn();
    }

    public void playerTurn() {
        Scanner scanner = new Scanner(System.in);
        while (true) {
            int[] FirstPoint = new int[]{playerLocation[0], playerLocation[1]};;
            System.out.println(Arrays.toString(playerLocation));
            listtoscreen();
            String nl = scanner.nextLine();
            Move mv = new Move();
            int[] sol = null;
            switch (nl) {
                case "w":
                    sol = mv.tryMove("y", -1, playerLocation, list, enemyLocation);
                    break;
                case "s":
                    sol = mv.tryMove("y", 1, playerLocation, list, enemyLocation);
                    break;
                case "a":
                    sol = mv.tryMove("x", -1, playerLocation, list, enemyLocation);
                    break;
                case "d":
                    sol = mv.tryMove("x", 1, playerLocation, list, enemyLocation);
                    break;
                default:
                    System.out.println("Invalid direction");
                    break;
            }
            if(sol == null){
                continue;
            }
            playerLocation[0] = sol[0];
            playerLocation[1] = sol[1];
            list[playerLocation[0]][playerLocation[1]] = 1;
            if (util.isEquallocation(playerLocation, goalLocation)) {
                System.out.println("You win");
                winnerset = true;
                break;
            }
            if(sol[2] == 1){
                list[FirstPoint[0]][FirstPoint[1]] = 0;
                break;
            }

        }
        if(!winnerset) {
            EnemyTurn();
        }
        else {
            System.out.println("You win");
        }
    }

    public void EnemyTurn(){
        Calculate cal = new Calculate(util.createboard(list), new int[]{playerLocation[0], playerLocation[1]}, new int[]{enemyLocation[0], enemyLocation[1]});
        cal.findsolve();
        List<String> Move = cal.findsolve();
        System.out.println("Player Win rate: "  + (-1) * cal.getScore());
        for(String s: Move) {
            int[] FirstPoint = new int[]{enemyLocation[0], enemyLocation[1]};;
            Move mv = new Move();
            int[] sol = null;
            if(Objects.equals(s, "w")){
                sol = mv.tryMove("y", -1, enemyLocation, list, playerLocation);
            }
            else if (Objects.equals(s, "s")){
                sol = mv.tryMove("y", 1, enemyLocation, list, playerLocation);
            }
            else if (Objects.equals(s, "a")){
                sol = mv.tryMove("x", -1, enemyLocation, list, playerLocation);
            }
            else if (Objects.equals(s, "d")){
                sol = mv.tryMove("x", 1, enemyLocation, list, playerLocation);
            }
            if(sol == null){
                continue;
            }
            enemyLocation[0] = sol[0];
            enemyLocation[1] = sol[1];
            list[enemyLocation[0]][enemyLocation[1]] = 1;
            if (util.isEquallocation(enemyLocation, goalLocation)) {
                System.out.println("You lose");
                winnerset = true;
                break;
            }
            if(sol[2] == 1){
                list[FirstPoint[0]][FirstPoint[1]] = 0;
                break;
            }

        }
        if(!winnerset) {
            playerTurn();
        }
    }

    private void resetboard(){
        list = new int[][]{
                {1, 1, 0, 0, 0, 1, 1}, // x
                {1, 0, 0, 0, 0, 0, 1},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0}, // 2-> goal
                {0, 0, 0, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 0, 1},
                {1, 1, 0, 0, 0, 1, 1}
                // y
        };
    }

    private void listtoscreen(){
        for(int i=0;i<list.length;i++){
            StringBuilder str = new StringBuilder();
            for(int j=0;j<list[i].length;j++){
                if(list[i][j] == 0){
                    if(i == goalLocation[0] && j == goalLocation[1]){
                        str.append("G");
                    }
                    else {
                        str.append("□");
                    }
                }
                else if(list[i][j] == 1){
                    if(i == playerLocation[0] && j == playerLocation[1]){
                        str.append("P");
                    }
                    else if(i == enemyLocation[0] && j == enemyLocation[1]){
                        str.append("E");
                    }
                    else {
                        str.append("■");
                    }
                }
                else if(list[i][j] == 2){
                    str.append("G");
                }
                str.append(" ");
            }
            System.out.println(str);
        }
    }
}
