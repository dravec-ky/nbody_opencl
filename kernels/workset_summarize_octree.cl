//Largely inspired by https://github.com/bneukom/gpu-nbody/blob/master/kernels/nbody/buildtree.cl
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#ifdef HAS__debug
    #define DEBUG(id,x,val) _debug[id*8+x] = val
#else 
    #define DEBUG(id,x,val) do {} while (0)
#endif

kernel void summarize_octree( 
#ifdef SINGLE_WORKGROUP
                 int num_worksets, 
                 global const int *worksets_begins, 
                 global const int *worksets_lengths, 
#else
                 int worksets_begin, 
                 int worksets_length, 
#endif
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
#ifndef SINGLE_WORKGROUP
    if(worksets_begin!=0) return;
#endif
    __local volatile int local_tree[MAX_LOCAL_SIZE * 8];

    const int NUM_BODIES = _cell_pos_length-1;
    const int NUM_NODES = NUM_BODIES*2+1;
    const int CELL_OFFSET = NUM_BODIES+1;
    const int NUM_CELLS = 8;
    const int MAX_LOOP = 1000;

    const int local_id = get_local_id(0);
    int idx = get_global_id(0);
    if(idx>NUM_BODIES) return;

    int bottom = *_bottom;
    int step_size = get_local_size(0)*get_num_groups(0);

    int node = (bottom) + idx;
    int is_cell;
    int missing_count = 0;

    int cell_body_count = 0;
    float cell_mass;
    float mass;
    float centerX,centerY,centerZ;

    //DEBUG(idx,3, step_size);
    int i=0;
    while((node < NUM_BODIES && node >= 0))
    {
        DEBUG(node,2,get_local_size(0));
        //new cell
        if(missing_count == 0)
        {
            cell_mass = 0.0f;
            centerX = 0.0f;
            centerY = 0.0f;
            centerZ = 0.0f;
            cell_body_count = 0;

            int used_child_idx = 0;

            #pragma unroll 8
            for (int child_idx = 0; child_idx < 8; child_idx++)
            {
                int child = _cell_tree[node * NUM_CELLS + child_idx];

                is_cell = child > NUM_BODIES;
                //Whether the child is occupied
                if(child>= 0)
                {
                    if(child_idx != used_child_idx)
                    {
                        _cell_tree[node * NUM_CELLS + child_idx] = -1;
                        _cell_tree[node * NUM_CELLS + used_child_idx] = child;
                    }

                    local_tree[MAX_LOCAL_SIZE * missing_count + local_id] = child;

                    if(is_cell)
                    {
                        mass = _cell_mass[child-CELL_OFFSET];
                    }
                    else
                    {
                        mass = _mass[child];
                    }
                    ++missing_count;
                    if(mass >= 0.0f)
                    {
                        --missing_count;

                        if(is_cell)
                        {
                            cell_body_count += _cell_body_count[child-CELL_OFFSET]-1;
                            centerX += _cell_pos[(child-CELL_OFFSET)*3] * mass;
                            centerY += _cell_pos[(child-CELL_OFFSET)*3+1] * mass;
                            centerZ += _cell_pos[(child-CELL_OFFSET)*3+2] * mass;
                        }
                        else
                        {
                            centerX += _pos[(child)*3] * mass;
                            centerY += _pos[(child)*3+1] * mass;
                            centerZ += _pos[(child)*3+2] * mass;
                        }
                        cell_mass += mass;
                    }
                    used_child_idx++;
                }
            }
            cell_body_count += used_child_idx;
        }

        if(missing_count != 0)
        {
            do
            {
                int child = local_tree[(missing_count-1) * MAX_LOCAL_SIZE + local_id];
                is_cell = child > NUM_BODIES;
                if(is_cell)
                {
                    mass = _cell_mass[child-CELL_OFFSET];
                }
                else
                {
                    mass = _mass[child];
                }
                if(mass>=0.0f)
                {
                    --missing_count;
                    if(is_cell)
                    {
                        cell_body_count += _cell_body_count[child-CELL_OFFSET]-1;

                        centerX += _cell_pos[(child-CELL_OFFSET)*3] * mass;
                        centerY += _cell_pos[(child-CELL_OFFSET)*3+1] * mass;
                        centerZ += _cell_pos[(child-CELL_OFFSET)*3+2] * mass;
                    }
                    else
                    {
                        centerX += _pos[(child)*3] * mass;
                        centerY += _pos[(child)*3+1] * mass;
                        centerZ += _pos[(child)*3+2] * mass;
                    }
                    cell_mass += mass;
                }
            }
            while(mass >= 0.0f && missing_count != 0);
        }
        if(missing_count == 0)
        {
            _cell_body_count[node] = cell_body_count;
            mass = 1.0 / cell_mass;
            _cell_pos[(node)*3] = centerX * mass;
            _cell_pos[(node)*3+1] = centerY * mass;
            _cell_pos[(node)*3+2] = centerZ * mass;

            _cell_mass[node] = cell_mass;
            node+=step_size;
            
        }
        i++;
    }
}