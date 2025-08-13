//Largely inspired by https://github.com/bneukom/gpu-nbody/blob/master/kernels/nbody/buildtree.cl
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#ifdef HAS__debug
    #define DEBUG(id,x,val) _debug[id*8+x] = val
#else
    #define DEBUG(id,x,val) do {} while (0)
#endif
// #define EPSILON (0.05f * 0.05f)
#define EPSILON (0.00001f)

#define MAX_DEPTH (32)
#define NUM_CELLS (8)
#define MAX_LOOP (100000)

global volatile float dst_sq_limit[MAX_DEPTH];

kernel void calculate_gravity( 
                int _pos_length, 
                 global float * _pos ,
                 int _cell_pos_length, 
                 global float * _cell_pos ,
                 int _mass_length, 
                 global float * _mass ,
                 int _cell_mass_length, 
                 global float * _cell_mass ,
                 int _cell_tree_length, 
                 global int * _cell_tree ,
                 int _cell_body_count_length, 
                 global int * _cell_body_count ,
#ifdef HAS__cell_start
                 int _cell_start_length, 
                 global int * _cell_start ,
#endif
#ifdef HAS__cell_sorted
                 int _cell_sorted_length, 
                 global int * _cell_sorted ,
#endif
                 int _g_acc_length, 
                 global float * _g_acc ,
                 int _bottom_length, 
                 global int * _bottom ,
                 int _radius_length, 
                 global float * _radius ,
#ifdef HAS__debug
                 int _debug_length, 
                 global float * _debug ,
#endif
#ifdef HAS__octree_depth
                 int _octree_depth_length, 
                 global int * _octree_depth ,
#endif
#ifdef HAS__octree_pos
                 int _octree_pos_length, 
                 global float * _octree_pos ,
#endif
#ifdef HAS__preview_id
                 int _preview_id_length, 
                 global int * _preview_id ,
#endif
#ifdef HAS__preview_switch
                 int _preview_switch_length, 
                 global int * _preview_switch ,
#endif
                 int _max_depth_length, 
                 global int * _max_depth ,
                 int _incr_length, 
                 global int * _incr ,
                 int _theta_length, 
                 global float * _theta ,
                 int _softening_length, 
                 global float * _softening ,
                 int _world_scale_length, 
                 global float * _world_scale,
                 int _mass_scale_length, 
                 global float * _mass_scale,
                 int _g_temp_length, 
                 global float * _g_temp 
)
{
    //to be input vars
    float THETA = *_theta;
    float SOFTENING = *_softening;
    float WORLD_SCALE = *_world_scale;
    float MASS_SCALE = *_mass_scale;
    
    const int NUM_BODIES = _cell_pos_length-1;
    const int NUM_NODES = NUM_BODIES*2+1;
    const int CELL_OFFSET = NUM_BODIES+1;
    #ifdef HAS__preview_id
    const int preview_id = *_preview_id;
    #endif

    int local_pos[MAX_DEPTH];
    int local_node[MAX_DEPTH];

    const int overall_max_depth = min(MAX_DEPTH,*_max_depth);

    const int local_id = get_local_id(0);

    int idx = get_global_id(0);
    float radius = *_radius;

	dst_sq_limit[0] = radius * radius / (THETA+0.00001);

    int i;
    if(local_id == 0)
    {
        for (i = 1; i < *_max_depth; ++i)
        {
            dst_sq_limit[i] = 0.25f*dst_sq_limit[i-1];
            dst_sq_limit[i-1] += SOFTENING;
        }
        dst_sq_limit[i-1] += SOFTENING;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(idx>NUM_BODIES) return;

    int offset = 0;
#ifdef HAS__cell_sorted
    int idx_body = _cell_sorted[idx];
#else
    int idx_body = idx;
#endif
    float posX = _pos[idx_body*3];
    float posY = _pos[idx_body*3+1];
    float posZ = _pos[idx_body*3+2];

    float own_mass = _mass[idx_body]*MASS_SCALE;

    float accX, accY, accZ, temperature;
    accX = accY = accZ = temperature = 0.0f;

    int tree_depth = NUM_NODES;

    int cur_node = offset;
    local_node[offset] = NUM_BODIES;
    local_pos[offset] = 0;

    int o = 0;
    while(cur_node >= offset)
    {
        int top;
        while((top = local_pos[cur_node]) < 8 && o < MAX_LOOP)
        {
            int child = _cell_tree[local_node[cur_node]*NUM_CELLS + top];
            local_pos[cur_node] = top + 1;
            //maybe not needed
            mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            if(child >=0 && child <= NUM_NODES)
            {
                float distX, distY, distZ, mass;
                int is_cell = child > NUM_BODIES;
                if(is_cell)
                {
                    distX = (_cell_pos[(child-CELL_OFFSET)*3] - posX);
                    distY = (_cell_pos[(child-CELL_OFFSET)*3+1] - posY);
                    distZ = (_cell_pos[(child-CELL_OFFSET)*3+2] - posZ);
                    mass = _cell_mass[child-CELL_OFFSET]*MASS_SCALE;                 
                }
                else
                {
                    
                    distX = (_pos[child*3] - posX);
                    distY = (_pos[child*3+1] - posY);
                    distZ = (_pos[child*3+2] - posZ);
                    mass = _mass[child]*MASS_SCALE;
                    #ifdef HAS__preview_id
                    if(idx_body == preview_id)
                    {
                        _preview_switch[child]++;
                    }
                    #endif
                }
                float dist_squared_base = distX*distX + distY*distY + distZ*distZ;
                float dist_squared = (dist_squared_base)*WORLD_SCALE*WORLD_SCALE + SOFTENING;
                if(!is_cell || dist_squared_base >= dst_sq_limit[cur_node-offset] || cur_node - offset >= overall_max_depth-1)
                {
                    float rdistance = rsqrt(dist_squared);
                    float f = own_mass * mass * rdistance * rdistance * rdistance;
                    accX += distX * f;
                    accY += distY * f;
                    accZ += distZ*WORLD_SCALE * f;
                    temperature += f;

                    #ifdef HAS__preview_id
                    if(is_cell && idx_body == preview_id)
                    {
                        const int cell_idx = child-CELL_OFFSET;
                        _preview_switch[cell_idx]+=2;
                    }
                    #endif
                }
                else
                {
                    cur_node++;
                    local_node[cur_node] = child-CELL_OFFSET;
                    local_pos[cur_node] = 0;

                    mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
                }
            }
            else
            {
                cur_node = max(offset, cur_node - 1);
                cur_node = min(MAX_DEPTH-1, cur_node);
            }
            o++;
        } 
        cur_node--;
    }
    _g_acc[idx_body*3] = accX;
    _g_acc[idx_body*3+1] = accY;
    _g_acc[idx_body*3+2] = accZ;
    _g_temp[idx_body] = temperature;
    
}