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

int main() {
    string a, b;
    cin >> a >> b;
    int x = 1;
    for (int i = 1; i < a.size(); ++i) {
        if (a[i] == b[i]) continue;
        if (a[i] == '8') {
            if (b[i] == '(') {
                x--;
            } else {
                x++;
            }
        } else if (a[i] == '(') {
            if (b[i] == '[') {
                x--;
            } else {
                x++;
            }
        } else {
            if (b[i] == '(') {
                x++;
            } else {
                x--;
            }
        }
    }
    if (x == 0) {
        cout << "TIE" << endl;
    } else if (x > 0) {
        cout << "TEAM 1 WINS" << endl;
    } else {
        cout << "TEAM 2 WINS" << endl;
    }
    return 0;
}
