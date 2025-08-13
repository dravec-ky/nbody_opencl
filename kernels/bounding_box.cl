//Code implemented according to https://github.com/bneukom/gpu-nbody/blob/master/kernels/nbody/
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#ifdef HAS__debug 
    #define DEBUG(id,x,val) _debug[id*8+x] = val
#else
    #define DEBUG(id,x,val) do {} while (0)
#endif

__global volatile float _minX[MAX_WORKGROUP_ID], _minY[MAX_WORKGROUP_ID],
		_minZ[MAX_WORKGROUP_ID];
__global volatile float _maxX[MAX_WORKGROUP_ID], _maxY[MAX_WORKGROUP_ID],
		_maxZ[MAX_WORKGROUP_ID];

//__attribute__ ((reqd_work_group_size(WORKGROUP_SIZE,1, 1)))
kernel void bounding_box( 
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
    __local volatile float localMinX[MAX_LOCAL_SIZE], localMinY[MAX_LOCAL_SIZE], localMinZ[MAX_LOCAL_SIZE];
	__local volatile float localMaxX[MAX_LOCAL_SIZE], localMaxY[MAX_LOCAL_SIZE], localMaxZ[MAX_LOCAL_SIZE];
    int workset_id = -1;
    float3 local_min;
    float3 local_max;
//BODY
    //SET MAIN IDS
    const int global_id = get_global_id(0);
    const int global_size = get_global_size(0);

    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    int sizex = _pos_length;
    int idx = get_global_id(0);
    if (idx >= _cell_pos_length)
        return;

   const int NUM_BODIES = _cell_pos_length-1;
   const int NUM_NODES = NUM_BODIES*2+1;
   const int NUM_CELLS = 8;

   #ifdef HAS__cell_start
   _cell_start[global_id] = -1;
   #endif

   DEBUG(idx,0,get_group_id(0));
   DEBUG(idx,1,MAX_WORKGROUP_ID);

    if(local_id == 0)
    {
      localMinX[0] = _pos[0];
		localMinY[0] = _pos[1];
		localMinZ[0] = _pos[2];
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

   localMinX[local_id] = localMaxX[local_id] = localMinX[0];
	localMinY[local_id] = localMaxY[local_id] = localMinY[0];
	localMinZ[local_id] = localMaxZ[local_id] = localMinZ[0];

   localMinX[local_id] = fmin(localMinX[local_id], _pos[global_id*3]);
   localMinY[local_id] = fmin(localMinY[local_id], _pos[global_id*3+1]);
   localMinZ[local_id] = fmin(localMinZ[local_id], _pos[global_id*3+2]);

   localMaxX[local_id] = fmax(localMaxX[local_id], _pos[global_id*3]);
   localMaxY[local_id] = fmax(localMaxY[local_id], _pos[global_id*3+1]);
   localMaxZ[local_id] = fmax(localMaxZ[local_id], _pos[global_id*3+2]);

   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

   for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2)
   {
		if (local_id < offset && global_id+offset<_cell_pos_length)
      {
			localMinX[local_id] = fmin(localMinX[local_id], localMinX[local_id + offset]);
			localMinY[local_id] = fmin(localMinY[local_id], localMinY[local_id + offset]);
			localMinZ[local_id] = fmin(localMinZ[local_id], localMinZ[local_id + offset]);

			localMaxX[local_id] = fmax(localMaxX[local_id], localMaxX[local_id + offset]);
			localMaxY[local_id] = fmax(localMaxY[local_id], localMaxY[local_id + offset]);
			localMaxZ[local_id] = fmax(localMaxZ[local_id], localMaxZ[local_id + offset]);
	   }
		barrier(CLK_LOCAL_MEM_FENCE);
	}

   if (local_id == 0)
   {
      _minX[group_id] = localMinX[0];
		_minY[group_id] = localMinY[0];
		_minZ[group_id] = localMinZ[0];

		_maxX[group_id] = localMaxX[0];
		_maxY[group_id] = localMaxY[0];
		_maxZ[group_id] = localMaxZ[0];

      //wait for completion
      mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
      //push results to other work items
      atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);

      const int last_block_id = MAX_WORKGROUP_ID-1;

      if (last_block_id == atom_inc(_incr))
      {
         DEBUG(idx,6,last_block_id);
         for (int i = 1; i <= last_block_id; ++i)
         {
            localMinX[0] = fmin(localMinX[0], _minX[i]);
				localMinY[0] = fmin(localMinY[0], _minY[i]);
				localMinZ[0] = fmin(localMinZ[0], _minZ[i]);

				localMaxX[0] = fmax(localMaxX[0], _maxX[i]);
				localMaxY[0] = fmax(localMaxY[0], _maxY[i]);
				localMaxZ[0] = fmax(localMaxZ[0], _maxZ[i]);
         }
         const float rootX = 0.5f * (localMinX[0] + localMaxX[0]);
			const float rootY = 0.5f * (localMinY[0] + localMaxY[0]);
			const float rootZ = 0.5f * (localMinZ[0] + localMaxZ[0]);

         *_radius = 0.5f * fmax(fmax(localMaxX[0]- localMinX[0], localMaxY[0] - localMinY[0]), localMaxZ[0] - localMinZ[0]);
			*_bottom = NUM_BODIES;
			*_incr = 0;

         _cell_pos[NUM_BODIES*3] = rootX;
        _cell_pos[NUM_BODIES*3+1] = rootY;
        _cell_pos[NUM_BODIES*3+2] = rootZ;
			_cell_mass[NUM_BODIES] = -1.0f;
#ifdef HAS__cell_start
			_cell_start[NUM_BODIES] = 0;
#endif
      _cell_body_count[NUM_BODIES] = 0;

#ifdef HAS__octree_depth
         _octree_depth[(NUM_BODIES)] = 1;
         _octree_pos[(NUM_BODIES)*3] = rootX;
         _octree_pos[(NUM_BODIES)*3+1] = rootY;
         _octree_pos[(NUM_BODIES)*3+2] = rootZ;
#endif

         #pragma unroll 8
			for (int i = 0; i < NUM_CELLS; i++)
            {
                _cell_tree[NUM_BODIES * NUM_CELLS + i] = -1;
                _cell_mass[NUM_BODIES * NUM_CELLS + i] = -1;
            }
      }
   }

}