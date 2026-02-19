#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <list>
#include <fstream>
#include <climits>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <bitset>
using namespace std;

vector<vector<int>> vec(3);
map<string, int> like;
int diff, ans, a, b, c;
int mat[10][10];

int cal() {
    int sum = 0, i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < vec[i].size(); j++) {
            for (k = 0; k < vec[i].size(); k++) {
                sum += mat[vec[i][j]][vec[i][k]];
            }
        }
    }
    return sum;
}

void count(int now) {
    if (now == 7) {
        if (vec[0].size() && vec[1].size() && vec[2].size()) {
            int x[3] = { a / vec[0].size(), b / vec[1].size(), c / vec[2].size() };
            sort(x, x + 3);
            if (x[2] - x[0] < diff) {
                diff = x[2] - x[0];
                ans = cal();
            } else if (x[2] - x[0] == diff) {
                ans = max(ans, cal());
            }
        }
        return;
    }
    for (int i = 0; i < 3; i++) {
        vec[i].push_back(now);
        count(now + 1);
        vec[i].pop_back();
    }
}

int main() {
    like["Anka"] = 0;
    like["Chapay"] = 1;
    like["Cleo"] = 2;
    like["Troll"] = 3;
    like["Dracul"] = 4;
    like["Snowy"] = 5;
    like["Hexadecimal"] = 6;
    int n, i;
    string name1, str, name2;
    while (scanf("%d", &n) != EOF) {
        diff = (1 << 31) - 1;
        for (i = 0; i < 3; i++) vec[i].clear();
        memset(mat, 0, sizeof(mat));
        for (i = 0; i < n; i++) {
            cin >> name1 >> str >> name2;
            mat[like[name1]][like[name2]]++;
        }
        cin >> a >> b >> c;
        count(0);
        cout << diff << " " << ans << endl;
    }
    return 0;
}
