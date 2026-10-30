package org.example;

import org.example.Learning.BitBoard;
import org.example.Learning.Calculate;
import org.example.Learning.SliderAI;

import java.util.*;

/**
 * 말 2개씩 두는 콘솔 게임.
 * - 한 턴에 말 하나만 골라 움직인다. 어느 말이든 골에 멈추면 이긴다.
 * - 다른 말(상대 말과 내 다른 말)이 있는 칸으로는 갈 수 없다.
 * - 둘 수 있는 수가 없으면 차례가 넘어간다.
 * - 같은 국면(누구 차례인지 포함)이 3번 나오면 무승부다.
 */
public class Game {

    private enum Outcome {MOVED, PASSED, WON, QUIT}

    private int[][] list;

    // 위쪽 두 귀퉁이가 플레이어, 아래쪽 두 귀퉁이가 AI (180도 돌리면 서로 겹치는 공정한 배치)
    private final int[][] playerPieces = {{0, 0}, {0, 6}};
    private final int[][] enemyPieces = {{6, 0}, {6, 6}};
    private final int[] goalLocation = new int[]{3, 3};

    private static final char[] PLAYER_MARK = {'1', '2'};
    private static final char[] ENEMY_MARK = {'X', 'Y'};

    // SliderAI.positionHash 기준으로 센 국면 횟수
    private final Map<Long, Integer> positions = new HashMap<>();

    public void start() {
        resetboard();
        Scanner scanner = new Scanner(System.in);
        boolean playerToMove = true;
        recordPosition(playerToMove);
        while (true) {
            Outcome outcome = playerToMove ? playerTurn(scanner) : EnemyTurn();
            if (outcome == Outcome.QUIT) {
                System.out.println("게임을 종료합니다.");
                return;
            }
            if (outcome == Outcome.WON) {
                listtoscreen();
                System.out.println(playerToMove ? "You win" : "You lose");
                return;
            }
            playerToMove = !playerToMove;
            if (recordPosition(playerToMove) >= 3) {
                listtoscreen();
                System.out.println("같은 국면이 3번 반복되어 무승부입니다.");
                return;
            }
        }
    }

    private Outcome playerTurn(Scanner scanner) {
        boolean[] movable = Calculate.movablePieces(list, playerPieces, enemyPieces);
        if (!any(movable)) {
            System.out.println("플레이어는 둘 수 있는 수가 없어 차례를 넘깁니다.");
            return Outcome.PASSED;
        }
        while (true) {
            listtoscreen();
            int idx = choosePiece(scanner, movable);
            if (idx < 0) {
                return Outcome.QUIT;
            }
            int[][] savedBoard = util.createboard(list);
            int[] savedLoc = playerPieces[idx].clone();
            int[] loc = playerPieces[idx];
            Move mv = new Move();
            while (true) {
                if (!scanner.hasNextLine()) {
                    return Outcome.QUIT;
                }
                String nl = scanner.nextLine().trim();
                if (nl.equals("r")) {
                    // 걷다가 막혔거나 마음이 바뀌면 이번 턴을 처음부터 다시 한다
                    list = savedBoard;
                    loc[0] = savedLoc[0];
                    loc[1] = savedLoc[1];
                    break;
                }
                int[] d = direction(nl);
                if (d == null) {
                    System.out.println("Invalid direction (w/a/s/d, r = 이번 턴 처음부터)");
                    continue;
                }
                int[] first = new int[]{loc[0], loc[1]};
                int[] sol = mv.tryMove(d[0], d[1], loc, list, othersOf(playerPieces, idx, enemyPieces));
                if (sol == null) {
                    continue;
                }
                loc[0] = sol[0];
                loc[1] = sol[1];
                list[loc[0]][loc[1]] = 1;
                if (sol[2] == 1) {
                    list[first[0]][first[1]] = 0;
                }
                if (util.isEquallocation(loc, goalLocation)) {
                    return Outcome.WON;
                }
                if (sol[2] == 1) {
                    return Outcome.MOVED;
                }
                listtoscreen();
            }
        }
    }

    private int choosePiece(Scanner scanner, boolean[] movable) {
        while (true) {
            System.out.print("움직일 말 (1/2): ");
            if (!scanner.hasNextLine()) {
                return -1;
            }
            String nl = scanner.nextLine().trim();
            int idx = nl.equals("1") ? 0 : nl.equals("2") ? 1 : -1;
            if (idx < 0) {
                System.out.println("1 또는 2를 입력하세요.");
            } else if (!movable[idx]) {
                System.out.println("그 말은 움직일 수 없습니다.");
            } else {
                return idx;
            }
        }
    }

    public static void clearScreen() {
        for (int i = 0; i < 80; i++)
            System.out.println();
    }

    private Outcome EnemyTurn() {
        boolean[] movable = Calculate.movablePieces(list, enemyPieces, playerPieces);
        if (!any(movable)) {
            System.out.println("AI는 둘 수 있는 수가 없어 차례를 넘깁니다.");
            return Outcome.PASSED;
        }
        Calculate cal = new Calculate(util.createboard(list), copyOf(playerPieces), copyOf(enemyPieces))
                .withGameHistory(positions);
        long beforeTime = System.currentTimeMillis();
        List<String> Move = cal.findsolve();
        long afterTime = System.currentTimeMillis(); // 코드 실행 후에 시간 받아오기
        long secDiffTime = (afterTime - beforeTime); //두 시간에 차 계산
        int[] pieceLoc = cal.getPieceLocation();
        int idx = pieceIndexAt(enemyPieces, pieceLoc);
        System.out.println("시간차이(m) : " + secDiffTime + ", 탐색 깊이: " + cal.getDepth());
        System.out.printf("AI 이동: 말 %c %s, Player Win rate: %.1f%%%n", ENEMY_MARK[idx], Move, cal.getPlayerWinRate() * 100);

        int[] loc = enemyPieces[idx];
        Move mv = new Move();
        for (String s : Move) {
            int[] first = new int[]{loc[0], loc[1]};
            int[] d = direction(s);
            int[] sol = mv.tryMove(d[0], d[1], loc, list, othersOf(enemyPieces, idx, playerPieces));
            if (sol == null) {
                throw new IllegalStateException("AI 이동을 실행할 수 없습니다: " + Move + " (말 " + ENEMY_MARK[idx] + ")");
            }
            loc[0] = sol[0];
            loc[1] = sol[1];
            list[loc[0]][loc[1]] = 1;
            if (sol[2] == 1) {
                list[first[0]][first[1]] = 0;
            }
            if (util.isEquallocation(loc, goalLocation)) {
                return Outcome.WON;
            }
            if (sol[2] == 1) {
                break;
            }
        }
        return Outcome.MOVED;
    }

    /** 이 국면을 기록하고 지금까지 나온 횟수(이번 포함)를 돌려준다. */
    private int recordPosition(boolean playerToMove) {
        long tiles = BitBoard.fromArray(list);
        long player = BitBoard.pieces(playerPieces);
        long enemy = BitBoard.pieces(enemyPieces);
        long key = playerToMove ? SliderAI.positionHash(tiles, player, enemy) : SliderAI.positionHash(tiles, enemy, player);
        return positions.merge(key, 1, Integer::sum);
    }

    private static int[] direction(String s) {
        switch (s) {
            case "w":
                return new int[]{0, -1};
            case "s":
                return new int[]{0, 1};
            case "a":
                return new int[]{-1, 0};
            case "d":
                return new int[]{1, 0};
            default:
                return null;
        }
    }

    /** idx 번 말을 뺀 모든 말의 좌표. */
    private static int[][] othersOf(int[][] mine, int idx, int[][] theirs) {
        List<int[]> others = new ArrayList<>();
        for (int i = 0; i < mine.length; i++) {
            if (i != idx) {
                others.add(mine[i]);
            }
        }
        others.addAll(Arrays.asList(theirs));
        return others.toArray(new int[0][]);
    }

    private static int[][] copyOf(int[][] pieces) {
        int[][] copy = new int[pieces.length][];
        for (int i = 0; i < pieces.length; i++) {
            copy[i] = new int[]{pieces[i][0], pieces[i][1]};
        }
        return copy;
    }

    private static int pieceIndexAt(int[][] pieces, int[] loc) {
        for (int i = 0; i < pieces.length; i++) {
            if (util.isEquallocation(pieces[i], loc)) {
                return i;
            }
        }
        throw new IllegalStateException("해당 위치에 말이 없습니다: " + Arrays.toString(loc));
    }

    private static boolean any(boolean[] flags) {
        for (boolean f : flags) {
            if (f) {
                return true;
            }
        }
        return false;
    }

    private void resetboard(){
        list = new int[][]{
                //0->blank space, 1->slider
                {1, 1, 0, 0, 0, 1, 1},
                {1, 0, 0, 0, 0, 0, 1},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 0, 1},
                {1, 1, 0, 0, 0, 1, 1}
        };
    }

    private char pieceMarkAt(int row, int col) {
        int[] loc = new int[]{row, col};
        for (int i = 0; i < playerPieces.length; i++) {
            if (util.isEquallocation(playerPieces[i], loc)) {
                return PLAYER_MARK[i];
            }
        }
        for (int i = 0; i < enemyPieces.length; i++) {
            if (util.isEquallocation(enemyPieces[i], loc)) {
                return ENEMY_MARK[i];
            }
        }
        return 0;
    }

    private void listtoscreen(){
        for(int i=0;i<list.length;i++){
            StringBuilder str = new StringBuilder();
            for(int j=0;j<list[i].length;j++){
                char mark = pieceMarkAt(i, j);
                if(list[i][j] == 0){
                    if(i == goalLocation[0] && j == goalLocation[1]){
                        str.append("G");
                    }
                    else {
                        str.append("□");
                    }
                }
                else if(list[i][j] == 1){
                    str.append(mark != 0 ? String.valueOf(mark) : "■");
                }
                else if(list[i][j] == 2){
                    str.append("G");
                }
                str.append(" ");
            }
            System.out.println(str);
        }
        System.out.println("플레이어 말: 1=" + Arrays.toString(playerPieces[0]) + " 2=" + Arrays.toString(playerPieces[1])
                + "  AI 말: X=" + Arrays.toString(enemyPieces[0]) + " Y=" + Arrays.toString(enemyPieces[1]));
    }
}
