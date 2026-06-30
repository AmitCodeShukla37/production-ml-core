#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int n = grid.size(), m = grid[0].size(), ans = 0;
        vector<vector<bool>>vis(n,vector<bool>(m,false));
        vector<int>dx = {0, 0, -1, 1};
        vector<int>dy = {1, -1, 0, 0};

        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(!vis[i][j] && grid[i][j] == '1'){
                    queue<pair<int,int>>q;
                    ans++;
                    q.push({i,j}); vis[i][j] = true;
                    while(!q.empty()){
                        int x = q.front().first, y = q.front().second; q.pop();
                        for(int k = 0; k < 4; k++){
                            int dxx = x + dx[k], dyy = y + dy[k];
                            if(dxx >= 0 && dxx < n && dyy >= 0 && dyy < m && !vis[dxx][dyy] && grid[dxx][dyy] == '1'){
                                q.push({dxx,dyy});
                                vis[dxx][dyy] = true;
                            }
                        }
                    }
                }
            }
        }
        
        return ans;
    }
};