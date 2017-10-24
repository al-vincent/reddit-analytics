import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
//import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
//import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

//=============================================================================
// MSc Project - Reddit PostCount MapReduce Program
//=============================================================================

// The Driver class is where Hadoop parameters can be set such as the
// Split size, mapper and reducer input/output Key, Value types and
// input/output paths.
//
// It is also the place where command-line arguments can be processed
// and Usage options can be printed to the terminal. 

public class SubRedditCountDriver extends Configured implements Tool {

	static { 
		Configuration.addDefaultResource("hdfs-default.xml"); 
		Configuration.addDefaultResource("hdfs-site.xml"); 
		Configuration.addDefaultResource("yarn-default.xml"); 
		Configuration.addDefaultResource("yarn-site.xml"); 
		Configuration.addDefaultResource("mapred-default.xml"); 
		Configuration.addDefaultResource("mapred-site.xml");
	}

	public static void printUsage(Tool tool, String extraArgsUsage) {
		System.err.printf("Usage: %s [genericOptions] %s\n\n",
				tool.getClass().getSimpleName(), extraArgsUsage);
		GenericOptionsParser.printGenericCommandUsage(System.err);
	}

	@Override
	public int run(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
		if (args.length != 2) {
			printUsage(this, "<input> <output>");
			return 1;
		}

		//Job job = Job.getInstance(new Configuration()); // this is a mistake; should pick up config which is already
		// set by GenericOptionParser and ToolRunner otherwise all command line options, 
		// such as -libjars, get ignored
		
		// Pick up configuration that has been set by GenericOptionParser 
		// and ToolRunner for further customisation if necessary
		
		Configuration config = getConf();
		FileSystem fs = FileSystem.get(config);
		
		// Here we use config to alter the number of reduce tasks.
		// This will be important when running the job on a cluster.
		/* ********************** WARNING!! **********************
		 * AV:  if running locally, this will split the output file
		 * into <nred> files (as it would on a cluster). Comment out
		 * to keep the output in a single file. 
		 * ********************** WARNING!! **********************/
		String nredString = config.get("mapreduce.job.reduces");
		int nred = Integer.parseInt(nredString);
		if (nred < 4)
			config.setInt("mapreduce.job.reduces", 4);
		
		// Get an instance of the Job class and use it to set various 
		// parameters for the MapReduce job we want to run.
		Job job = Job.getInstance(config);
		job.setJarByClass(getClass());

		// Delete old output if necessary
		Path outPath = new Path(args[1]);
		if (fs.exists(outPath)) 
			fs.delete(outPath, true);
		
		// Input and output paths are set from the command line arguments
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job,outPath);
		//LazyOutputFormat.setOutputFormatClass(job, TextOutputFormat.class);

		// Set Mapper, Combiner and Reducer classes
		job.setMapperClass(SubRedditCountMapper.class);
		job.setCombinerClass(SubRedditCountReducer.class);		
		job.setReducerClass(SubRedditCountReducer.class);

		// Set output key and value types for the mapper,
		// that is, the intermediate key/value types
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);

		// Output key and value types for the reducers
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		// Start the job and wait until it is completed.
		// Once completed, return its exit code (0 or 1).
		return job.waitForCompletion(true) ? 0 : 1;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("Running...");
		// You have to always use ToolRunner to execute your driver program
		// to ensure configuration parameters and the command line options 
		// are properly picked up and processed. 
		int exitCode = ToolRunner.run(new SubRedditCountDriver(), args);
		System.out.println("Complete");
		System.out.println("Exit code: " + exitCode);
		System.exit(exitCode);
	}

}
