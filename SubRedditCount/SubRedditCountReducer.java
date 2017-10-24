//=============================================================================
//CS5234 -- Distributed WordCount MapReduce Program -- The Reducer
//=============================================================================

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class SubRedditCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

	// You can put instance variables here to store state between iterations of
	// the reduce task.
	private IntWritable result = new IntWritable();	

	// The setup method. Anything in here will be run exactly once before the
	// beginning of the reduce task.
	public void setup(Context context) throws IOException, InterruptedException {
		//System.out.println("Reducing...");
	}
	
	// The reducer method
	public void reduce(Text key, Iterable<IntWritable> values, Context context)
			throws IOException, InterruptedException {
		// You can emit output records using
		// context.write(new Text("some string"), new Text("some other string"));
		// Note: you can use Context methods in the setup and cleanup as well.		
		int sum = 0;
		for (IntWritable val : values) {
			// accumulate the result in the 'sum' variable
			sum += val.get(); 
		}
		
		// Set and output the result
		this.result.set(sum);
		//System.out.println("Key: " + key + ", value: " + result.toString());
		//System.out.println("<key> class: " + key.getClass() + ", <result> class: " + result.getClass());
		context.write(key, this.result);		
	}
	
	// The cleanup method. Anything in here will be run exactly once after the
	// end of the reduce task.
	public void cleanup(Context context) throws IOException, InterruptedException {
		//System.out.println("Reduce Complete");
	}
}