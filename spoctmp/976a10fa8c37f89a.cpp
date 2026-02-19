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
    int count = 0;
    int arr[4];
    for (int i = 0; i < 4; ++i) {
        std::cin >> arr[i];
    }
    for (int i = 0; i < 3; ++i) {
        if (arr[i] != 0) {
            for (int j = i + 1; j < 4; ++j) {
                if (arr[i] == arr[j] && arr[j] != 0) {
                    ++count;
                    arr[j] = 0;
                }
            }
        }
    }
    std::cout << count << std::endl;
    return 0;
}
